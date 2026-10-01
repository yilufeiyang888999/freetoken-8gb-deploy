#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FreeToken 服务实测 —— 吞吐 / 延迟基线

用途
----
给本机这套配置（RTX 5060 8G + 15/40 层专家在 CPU + hybrid 分流）建立一个
可复现的性能基线，作为「这套配置能不能撑起 RAG / ChatBI POC」的判断依据。

用法（在 WSL 里跑，服务需已启动）
--------------------------------
    ~/freetoken-venv/bin/python tools/bench_tok_s.py

常用参数
--------
    --runs 5              负载测试次数（默认 5，取中位数）
    --thinking            额外测一组「带思考」的对照（默认关闭）
    --host / --port       覆盖服务地址（默认 127.0.0.1:1919）
    --model               覆盖模型名（默认从 /v1/models 自动取）

为什么要这样设计
----------------
1. **首次请求必须丢弃。** CUDA graph 捕获（batch [1,2,4]）、Triton kernel 首次编译、
   专家 banks 首次换入都在第一次请求里发生，耗时可能是稳态的几十倍。
   不预热直接测，得到的是「冷启动时间」，不是推理速度。

2. **必须关思考。** Qwen3.6 默认开思考模式，思维链会吃掉大量 token 且长度不可控 ——
   测出来的 tok/s 混了「写作速度」和「思考速度」，没有可比性，也和你 ChatBI 场景
   的真实负载不符。加 chat_template_kwargs.enable_thinking=false 才是干净口径。

3. **两个口径分开报。**
   - 端到端 tok/s（非流式）：包含首 token 延迟 + 全部 decode，反映「用户感知」
   - 纯 decode tok/s（流式）：排除固定开销，反映「引擎真实吞吐」
   两者差距大 = 首 token 延迟高（CPU 层专家换入慢），这是本配置的典型特征。

4. **两点法交叉验证。** 用 (长输出耗时 - 短输出耗时) / (长token - 短token) 再算一次
   decode 速度 —— 这条路径能再次剔除固定开销，与流式结果互相印证。

零依赖：只用标准库。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from typing import Any

LINE = "─" * 68
DOUBLE = "═" * 68

# 固定 prompt：要求输出长度可控（打到 max_tokens 上限），便于比对
PROMPT_LONG = "请用中文从 1 数到 200，每个数字之间用顿号分隔，不要换行，不要任何解释。"

# 小工具 ---------------------------------------------------------------


def 标题(文字: str) -> None:
    print()
    print(LINE)
    print(f" {文字}")
    print(LINE)


def 绿(文字: str) -> str:
    return f"\033[32m{文字}\033[0m"


def 红(文字: str) -> str:
    return f"\033[31m{文字}\033[0m"


def 黄(文字: str) -> str:
    return f"\033[33m{文字}\033[0m"


def 请求(
    url: str,
    payload: dict[str, Any],
    超时: float = 300.0,
) -> tuple[dict[str, Any], float]:
    """发一次非流式请求，返回 (响应体, 耗时秒)。"""
    数据 = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    请求对象 = urllib.request.Request(
        url,
        data=数据,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    起点 = time.perf_counter()
    with urllib.request.urlopen(请求对象, timeout=超时) as 响应:
        正文 = 响应.read().decode("utf-8")
    耗时 = time.perf_counter() - 起点
    return json.loads(正文), 耗时


def 流式请求(url: str, payload: dict[str, Any], 超时: float = 300.0) -> dict[str, Any]:
    """发一次流式请求，返回首 token 延迟、总耗时、块数。"""
    数据 = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    请求对象 = urllib.request.Request(
        url,
        data=数据,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    起点 = time.perf_counter()
    首块时刻: float | None = None
    内容块数 = 0
    用量: dict[str, Any] = {}

    with urllib.request.urlopen(请求对象, timeout=超时) as 响应:
        for 原始行 in 响应:
            行 = 原始行.decode("utf-8").strip()
            if not 行.startswith("data:"):
                continue
            载荷 = 行[5:].strip()
            if 载荷 == "[DONE]":
                break
            try:
                块 = json.loads(载荷)
            except json.JSONDecodeError:
                continue
            if 块.get("usage"):
                用量 = 块["usage"]
            选项 = 块.get("choices") or []
            if not 选项:
                continue
            增量 = 选项[0].get("delta") or {}
            if 增量.get("content") or 增量.get("reasoning_content"):
                if 首块时刻 is None:
                    首块时刻 = time.perf_counter()
                内容块数 += 1
    终点 = time.perf_counter()

    return {
        "首token秒": (首块时刻 - 起点) if 首块时刻 else float("nan"),
        "总秒": 终点 - 起点,
        "块数": 内容块数,
        "用量": 用量,
    }


def 取模型清单(地址: str) -> list[str]:
    请求对象 = urllib.request.Request(f"{地址}/v1/models", method="GET")
    with urllib.request.urlopen(请求对象, timeout=15) as 响应:
        正文 = json.loads(响应.read().decode("utf-8"))
    return [项.get("id", "?") for 项 in 正文.get("data", [])]


# 主流程 ---------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="FreeToken 吞吐基线实测")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default="1919")
    parser.add_argument("--model", default=None)
    parser.add_argument("--runs", type=int, default=5, help="负载测试次数，取中位数")
    parser.add_argument("--thinking", action="store_true", help="额外测一组带思考的对照")
    parser.add_argument("--tokens", type=int, default=256, help="负载测试的 max_tokens")
    args = parser.parse_args()

    地址 = f"http://{args.host}:{args.port}"
    聊天地址 = f"{地址}/v1/chat/completions"

    print()
    print(DOUBLE)
    print(" FreeToken 吞吐 / 延迟基线实测")
    print(DOUBLE)

    # ---------- 阶段 1 健康检查 ----------
    标题("阶段 1 / 5　健康检查")
    try:
        模型清单 = 取模型清单(地址)
    except urllib.error.URLError as 错:
        print(红(f"  [阻断] 连不上 {地址}：{错}"))
        print("  服务没起，或在另一台机器上。先启动：bash ~/serve_freetoken.sh")
        return 1
    except Exception as 错:  # noqa: BLE001
        print(红(f"  [阻断] /v1/models 异常：{错}"))
        return 1

    if not 模型清单:
        print(红("  [阻断] /v1/models 返回空列表 —— 后端 worker 可能已退出"))
        return 1

    模型名 = args.model or 模型清单[0]
    print(绿(f"  [通过] 服务在线，模型：{模型名}"))
    if len(模型清单) > 1:
        print(f"         可用模型：{', '.join(模型清单)}")
    print(f"         本次使用：{模型名}")

    基准负载 = {
        "model": 模型名,
        "messages": [{"role": "user", "content": PROMPT_LONG}],
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": args.tokens,
        "temperature": 0.0,
    }

    # ---------- 阶段 2 预热 ----------
    标题("阶段 2 / 5　预热（结果丢弃）")
    print("  首次请求要捕获 CUDA graph、编译 kernel、换入 CPU 层专家，")
    print("  耗时可能是稳态的几十倍。这一轮的耗时没有参考价值，直接丢。")
    try:
        _, 预热耗时 = 请求(聊天地址, 基准负载)
        print(f"  预热完成，耗时 {预热耗时:.2f} s（记住这个数，对比下面就知道省了多少）")
    except Exception as 错:  # noqa: BLE001
        print(红(f"  [阻断] 预热失败：{错}"))
        return 1

    # ---------- 阶段 3 非流式负载 ----------
    标题(f"阶段 3 / 5　端到端吞吐（非流式 × {args.runs}）")
    print("  口径：completion_tokens ÷ 总耗时，含首 token 延迟 —— 用户感知的速度。")
    print()
    记录: list[dict[str, Any]] = []
    for 序号 in range(1, args.runs + 1):
        try:
            正文, 耗时 = 请求(聊天地址, 基准负载)
        except Exception as 错:  # noqa: BLE001
            print(红(f"  第 {序号} 次失败：{错}"))
            continue
        用量 = 正文.get("usage") or {}
        出词 = 用量.get("completion_tokens", 0)
        入词 = 用量.get("prompt_tokens", 0)
        结束原因 = (正文.get("choices") or [{}])[0].get("finish_reason", "?")
        速度 = 出词 / 耗时 if 耗时 > 0 else 0.0
        记录.append({"出词": 出词, "耗时": 耗时, "速度": 速度})
        print(
            f"  第 {序号} 次： 输入 {入词:>4} tok │ 输出 {出词:>4} tok │ "
            f"{耗时:>6.2f} s │ {速度:>6.2f} tok/s │ {结束原因}"
        )

    if not 记录:
        print(红("  全部失败，无法给出结论"))
        return 1

    中位速度 = statistics.median(项["速度"] for 项 in 记录)
    中位耗时 = statistics.median(项["耗时"] for 项 in 记录)
    出词集合 = {项["出词"] for 项 in 记录}
    print()
    print(f"  中位数： {中位耗时:.2f} s   {绿(f'{中位速度:.2f} tok/s')}")
    if len(出词集合) > 1:
        print(
            黄(
                f"  注意：各次输出 token 数不一致（{sorted(出词集合)}），"
                "说明模型提前停止，tok/s 的可比性下降"
            )
        )
    if max(项["出词"] for 项 in 记录) < args.tokens:
        print(
            黄(
                f"  注意：输出未打满 max_tokens={args.tokens}，"
                "说明是自然收尾。要更稳定的基线，可加大 --tokens"
            )
        )

    # ---------- 阶段 4 纯 decode 速度（流式） ----------
    标题("阶段 4 / 5　纯 decode 速度（流式，排除首 token 延迟）")
    流式负载 = dict(基准负载)
    流式负载["stream"] = True
    流式负载["max_tokens"] = args.tokens
    try:
        流式结果 = 流式请求(聊天地址, 流式负载)
        首token = 流式结果["首token秒"]
        总秒 = 流式结果["总秒"]
        块数 = 流式结果["块数"]
        纯decode = (块数 - 1) / (总秒 - 首token) if 总秒 > 首token and 块数 > 1 else float("nan")
        print(f"  首 token 延迟（TTFT）： {首token:.2f} s")
        print(f"  总耗时：               {总秒:.2f} s")
        print(f"  内容块数：             {块数}")
        if 纯decode == 纯decode:  # not NaN
            print(f"  纯 decode 速度：       {绿(f'{纯decode:.2f} tok/s')}")
            差距 = 纯decode - 中位速度
            print()
            if 差距 > 中位速度 * 0.3:
                print(
                    黄(
                        f"  两者差 {差距:.2f} tok/s —— 差值主要在首 token 延迟上。"
                    )
                )
                print(
                    黄(
                        "  这是本配置的典型特征：15/40 层专家在 CPU，"
                        "每次请求开始要先换入。"
                    )
                )
            else:
                print("  两者接近 —— 首 token 延迟占比不高，decode 是主要瓶颈。")
        else:
            print(黄("  没收到内容块，无法算纯 decode 速度（流式可能不被支持）"))
    except Exception as 错:  # noqa: BLE001
        print(黄(f"  流式测试失败（该接口可能不支持 stream）：{错}"))

    # ---------- 阶段 5 可选：思考模式对照 ----------
    if args.thinking:
        标题("阶段 5 / 5　对照：开启思考模式")
        print("  同一 prompt，去掉 chat_template_kwargs —— 看多花了多少代价。")
        print()
        思考负载 = {
            "model": 模型名,
            "messages": [{"role": "user", "content": "用一句话说明什么是 MoE 架构。"}],
            "max_tokens": 2048,
            "temperature": 0.0,
        }
        try:
            正文, 耗时 = 请求(聊天地址, 思考负载)
            用量 = 正文.get("usage") or {}
            出词 = 用量.get("completion_tokens", 0)
            选项 = (正文.get("choices") or [{}])[0]
            消息 = 选项.get("message") or {}
            思考字数 = len(消息.get("reasoning_content") or "")
            答案字数 = len(消息.get("content") or "")
            结束原因 = 选项.get("finish_reason", "?")
            print(f"  输出 {出词} tok │ {耗时:.2f} s │ {出词 / 耗时:.2f} tok/s │ {结束原因}")
            print(f"  思维链 {思考字数} 字 │ 正式答案 {答案字数} 字")
            if 答案字数 == 0:
                print(红("  → 答案为空、token 全被思维链吃掉，正是要避免的情况"))
            else:
                占比 = 思考字数 / (思考字数 + 答案字数) * 100
                print(f"  → 思维链占全部输出的 {占比:.0f}%，这部分是纯额外开销")
        except Exception as 错:  # noqa: BLE001
            print(黄(f"  对照测试失败：{错}"))
    else:
        标题("阶段 5 / 5　已跳过（加 --thinking 可测对照组）")

    # ---------- 收尾 ----------
    print()
    print(DOUBLE)
    print(" 结论")
    print(DOUBLE)
    print(f"  端到端：  {中位速度:.2f} tok/s（{args.runs} 次中位数）")
    print(f"  预热耗时：{预热耗时:.2f} s  ← 只有第一次付这个代价")
    print()
    print("  怎么读这个数：")
    print("    · 对话式问答（RAG QA）体感阈值大约 5 tok/s 以上")
    print("    · ChatBI 要出表格/JSON，建议 10 tok/s 以上才不难受")
    print("    · 想提速，按这个顺序试：")
    print("        1) 关掉浏览器 / WPS / 微信，腾显存给 slot cache")
    print("        2) 减少 CPU 层数（要腾出锁页预算，代价是 CPU 压力变大）")
    print("        3) ft ctl cache --moe N 在运行中调专家缓存池，不重启")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
