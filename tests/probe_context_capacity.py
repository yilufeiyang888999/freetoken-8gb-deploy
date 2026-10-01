#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
探测 FreeToken 服务的**真实**上下文容量 —— 8268 是硬上限还是可增长？

背景（三个互相矛盾的观测）:
    启动日志      Allocating 8268 tokens for KV cache, K + V = 0.16 GiB
    /v1/models    max_model_len = 16384           ← 服务承诺能吃 16384
    /v1/cache/   limit s.kv_tokens.max = 182932   ← 理论可涨到 18 万

关键认识: cache_budget_bytes（3.49 GiB）是 KV / MoE / Mamba **共享**的预算池。
    num_pages=8268 只是启动时的切分点, 不是物理硬上限。
    交叉验证: 预算 ÷ kv_per_token     = 182932  == limits.kv_tokens.max
              预算 ÷ moe_per_expert  = 2110    == limits.moe_experts.max
              预算 ÷ mamba_per_slot  = 58.2    == limits.mamba_slots.max
    三个 max 全部精确命中「整个池只给这一种资源」—— 共享池模型成立。

本脚本要回答的问题:
    Q1  发一个 prompt > 8268 token 的请求, 服务会失败吗?
    Q2  如果成功, 是自动扩容（num_pages 变大 / last_rebuild 被触发）还是别的原因?
    Q3  真实断点在哪?

方法: 阶梯式发长 prompt（max_tokens=1 + 关思考, 把 decode 压到最小,
      耗时几乎全是 prefill）。每步前后抓 /v1/cache/status 对比几何变化。

用法:
    python3 tests/probe_context_capacity.py
    python3 tests/probe_context_capacity.py --targets 8000,9000,12000
    python3 tests/probe_context_capacity.py --base http://127.0.0.1:1919

零第三方依赖（urllib）。
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

# ------------------------------------------------------------------ 填充语料
# 用真实的中文技术文本, 避免重复字符导致的 tokenizer 边界异常
填充源 = (
    "制造业数字化转型的核心在于把供应链上每一个环节的数据打通。"
    "供应商准入需要核验营业执照、开户许可与质量体系认证。"
    "采购寻源要匹配料号、规格、交期与价格条款。"
    "到料检验记录批次、抽检比例与判定结果。"
    "库存管理必须保证账实一致，任何差异都要能追溯到原始单据。"
    "生产领料与完工入库构成成本归集的原始凭证，月末结转直接关系毛利。"
    "应付对账最终落到发票、入库单与合同三者匹配，差一分钱都得查到底。"
)

默认目标 = [4000, 8000, 9000, 12000, 16000]


def 取状态(基址: str, 超时: float = 10.0):
    """抓 /v1/cache/status 的关键几何字段。失败返回 None。"""
    try:
        with urllib.request.urlopen(基址 + "/v1/cache/status", timeout=超时) as 响应:
            数据 = json.loads(响应.read().decode("utf-8"))
    except Exception:
        return None
    几何 = 数据.get("geometry", {})
    return {
        "state": 数据.get("state"),
        "last_rebuild": 数据.get("last_rebuild"),
        "num_pages": 几何.get("num_pages"),
        "moe_cache_size": 几何.get("moe_cache_size"),
        "num_mamba_slots": 几何.get("num_mamba_slots"),
        "kv_tokens_max": (几何.get("limits") or {}).get("kv_tokens", {}).get("max"),
    }


def 取模型名(基址: str, 超时: float = 10.0) -> str:
    with urllib.request.urlopen(基址 + "/v1/models", timeout=超时) as 响应:
        数据 = json.loads(响应.read().decode("utf-8"))
    return 数据["data"][0]["id"]


def 发送(基址: str, 模型: str, 文本: str, 超时: float, 关思考: bool = True):
    """
    发一次非流式请求。返回 (ok, prompt_tokens, completion_tokens, 耗时秒, 错误文本)。
    max_tokens=1 让耗时几乎等于 prefill。
    """
    载荷 = {
        "model": 模型,
        "messages": [{"role": "user", "content": 文本}],
        "max_tokens": 1,
        "stream": False,
    }
    if 关思考:
        # OpenAI 协议下的首选写法（FreeToken 同时认 chat_template_kwargs）
        载荷["thinking"] = {"type": "disabled"}

    数据 = json.dumps(载荷, ensure_ascii=False).encode("utf-8")
    请求 = urllib.request.Request(
        基址 + "/v1/chat/completions",
        data=数据,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    起点 = time.time()
    try:
        with urllib.request.urlopen(请求, timeout=超时) as 响应:
            正文 = json.loads(响应.read().decode("utf-8"))
        耗时 = time.time() - 起点
        用量 = 正文.get("usage") or {}
        return (
            True,
            用量.get("prompt_tokens"),
            用量.get("completion_tokens"),
            耗时,
            "",
        )
    except urllib.error.HTTPError as 错:
        耗时 = time.time() - 起点
        try:
            细节 = 错.read().decode("utf-8", "replace")[:400]
        except Exception:
            细节 = ""
        return (False, None, None, 耗时, f"HTTP {错.code}: {细节}")
    except Exception as 错:
        return (False, None, None, time.time() - 起点, f"{type(错).__name__}: {错}")


def 校准(基址: str, 模型: str, 超时: float) -> float:
    """返回「每字符约等于多少 token」，用于按目标长度生成填充文本。"""
    样本 = 填充源 * 10
    ok, pt, _, _, 错 = 发送(基址, 模型, 样本, 超时)
    if not ok or not pt:
        raise SystemExit(f"校准请求失败，无法继续：{错}")
    比例 = pt / len(样本)
    print(f"  校准：{len(样本)} 字符 → {pt} token（{比例:.4f} token/字符）")
    return 比例


def main() -> int:
    解析器 = argparse.ArgumentParser(
        description="探测 FreeToken 真实上下文容量（阶梯式长 prompt 压力测试）"
    )
    解析器.add_argument("--base", default="http://127.0.0.1:1919", help="服务地址")
    解析器.add_argument(
        "--targets",
        default=",".join(str(x) for x in 默认目标),
        help="目标 prompt token 数，逗号分隔",
    )
    解析器.add_argument("--timeout", type=float, default=300.0, help="单请求超时秒数")
    解析器.add_argument("--keep-thinking", action="store_true", help="不注入关思考")
    参数 = 解析器.parse_args()

    目标 = [int(x) for x in 参数.targets.split(",") if x.strip()]

    print("═" * 72)
    print(" FreeToken 上下文容量探测")
    print("═" * 72)

    基线 = 取状态(参数.base)
    if 基线 is None:
        print(f"✗ 连不上 {参数.base}/v1/cache/status —— 服务没起？")
        return 2
    print(f"  服务        {参数.base}")
    print(f"  状态        {基线['state']}")
    print(f"  基线几何    num_pages={基线['num_pages']}  "
          f"moe_cache={基线['moe_cache_size']}  mamba_slots={基线['num_mamba_slots']}")
    print(f"  KV 上限     {基线['kv_tokens_max']} token")
    print(f"  last_rebuild {基线['last_rebuild']}")
    print()

    模型 = 取模型名(参数.base)
    print(f"  模型        {模型}")
    比例 = 校准(参数.base, 模型, 参数.timeout)
    print()

    表头 = f"{'目标tok':>8} {'实际tok':>8} {'结果':>6} {'耗时':>9} {'num_pages':>10} {'rebuild':>12}  备注"
    print(表头)
    print("─" * 72)

    结果 = []
    断点 = None
    for 目标值 in 目标:
        字符数 = max(1, int(目标值 / 比例))
        文本 = (填充源 * (字符数 // len(填充源) + 1))[:字符数]

        前 = 取状态(参数.base)
        ok, pt, ct, 耗时, 错 = 发送(参数.base, 模型, 文本, 参数.timeout,
                                   关思考=not 参数.keep_thinking)
        后 = 取状态(参数.base)

        页变化 = ""
        if 前 and 后 and 前["num_pages"] != 后["num_pages"]:
            页变化 = f"  ← num_pages {前['num_pages']}→{后['num_pages']}"
        重构 = ""
        if 前 and 后 and 前["last_rebuild"] != 后["last_rebuild"]:
            重构 = f"  ★ 触发 rebuild: {后['last_rebuild']}"

        页显示 = str(后["num_pages"]) if 后 else "?"
        重构显示 = str(后["last_rebuild"]) if 后 else "?"

        if ok:
            备注 = f"OK{页变化}{重构}"
        else:
            备注 = 错.replace("\n", " ")[:70]
            if 断点 is None:
                断点 = 目标值

        print(
            f"{目标值:>8} {str(pt if pt else '-'):>8} "
            f"{'✓' if ok else '✗':>6} {耗时:>8.1f}s {页显示:>10} {重构显示:>12}  {备注}"
        )
        结果.append({
            "目标": 目标值, "实际": pt, "成功": ok,
            "耗时": round(耗时, 2), "错误": 错,
            "num_pages_前": 前["num_pages"] if 前 else None,
            "num_pages_后": 后["num_pages"] if 后 else None,
        })

    print()
    print("═" * 72)
    print(" 判读")
    print("═" * 72)

    成功项 = [r for r in 结果 if r["成功"]]
    最大成功 = max((r["实际"] or 0) for r in 成功项) if 成功项 else 0
    扩过容 = any(
        r["num_pages_后"] and r["num_pages_前"] and r["num_pages_后"] > r["num_pages_前"]
        for r in 结果
    )

    print(f"  本次最大成功 prompt    {最大成功} token")
    print(f"  启动时 KV 切分点       {基线['num_pages']} token")
    if 断点:
        print(f"  首次失败于             {断点} token（目标值）")
    else:
        print("  首次失败于             未出现 —— 所有档位都吃下了")

    if 最大成功 > 基线["num_pages"]:
        print(f"  ★ 超过启动切分点 {最大成功 - 基线['num_pages']} token 仍然成功 "
              f"→ **8268 不是硬上限**")
    if 扩过容:
        print("  ★ 观察到 num_pages 自动增长 → 服务会按需从共享池给 KV 扩容")
    elif 最大成功 > 基线["num_pages"]:
        print("  · 未见 num_pages 变化 → 可能是请求期间临时分配、请求后立即归还")

    末尾 = 取状态(参数.base)
    if 末尾:
        print(f"  收尾几何              num_pages={末尾['num_pages']}  "
              f"moe={末尾['moe_cache_size']}  mamba={末尾['num_mamba_slots']}")
        if 末尾["num_pages"] != 基线["num_pages"]:
            print(f"  ⚠ 几何未复原（{基线['num_pages']} → {末尾['num_pages']}），"
                  f"MoE 命中率可能已被削减，建议重启服务恢复")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
