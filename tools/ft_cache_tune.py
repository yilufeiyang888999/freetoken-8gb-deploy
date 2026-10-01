#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FreeToken 缓存几何调优 —— 用闲置的并发槽去换上下文长度。

为什么需要它
------------
8GB 卡上，KV cache / MoE slot cache / Mamba slot 三块**共享一个固定的显存池**
（本机实测约 3.43 GiB）。预算约束是：

    可用(KV + MoE) = 总池 − mamba槽数 × 每槽字节

出厂切分 `mamba=24 / MoE=1108 / KV=8268` 恰好把池子用光 ——
所以即使启动时传了 `--max-seq-len-override 16384`（/v1/models 也如实报 16384），
**实际能吃的 prompt 只有 8268 token**，超过直接 400。

要让上下文真正到达 16384，必须从别处割让。三块的职责不同：

    Mamba slot  = 并发序列上限（每个活动请求占一个）→ 单用户根本用不满
    MoE cache   = 常驻显存的专家数 → 影响 decode 速度
    KV pages    = 上下文容量 → 本次要加的就是它

**最该让位的是 Mamba 槽。** 实测把 mamba 从 24 降到 21，KV 就能从 8268 涨到
16384，而纯 decode 速度没有可测变化（25.2~26.0 tok/s 各配置一致）。

实测数据（RTX 5060 Laptop 8G / Qwen3.6-35B-A3B-NVFP4）
------------------------------------------------------
    KV 8268  → prefill 8911 token 直接 400
    KV 16384 → prefill 15826 token 通过
    代价：并发上限 24→21；tok/s 无变化
    另测：MoE 1108→1290 无提速（说明 1108 已饱和，别再折腾这块）

用法
----
    python3 ft_cache_tune.py                     # 看现状 + 给出建议（不改动）
    python3 ft_cache_tune.py --context 16384     # 按目标上下文自动调
    python3 ft_cache_tune.py --context 16384 --moe 1108 --mamba 21
    python3 ft_cache_tune.py --restore           # 恢复出厂几何
    python3 ft_cache_tune.py --wait              # 服务忙时轮询等待（默认直接退出）

零第三方依赖。服务空闲时重建只需 1~6 秒。
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request

基址默认 = "http://127.0.0.1:1919"
出厂 = {"num_pages": 8268, "moe_cache_size": 1108, "num_mamba_slots": 24}
GiB = 1024 ** 3
MiB = 1024 ** 2


# ------------------------------------------------------------------ HTTP 小工具
def 取状态(基址: str):
    """返回 (cache_status dict, 错误文本)。"""
    try:
        with urllib.request.urlopen(基址 + "/v1/cache/status", timeout=10) as 响应:
            return json.loads(响应.read()), None
    except Exception as 错:
        return None, f"{type(错).__name__}: {错}"


def 取统计(基址: str):
    try:
        with urllib.request.urlopen(基址 + "/v1/stats", timeout=10) as 响应:
            return json.loads(响应.read()), None
    except Exception as 错:
        return None, f"{type(错).__name__}: {错}"


def 取模型(基址: str):
    try:
        with urllib.request.urlopen(基址 + "/v1/models", timeout=10) as 响应:
            数据 = json.loads(响应.read())
        首个 = 数据["data"][0]
        return 首个.get("max_model_len") or 首个.get("context_length"), None
    except Exception as 错:
        return None, f"{type(错).__name__}: {错}"


def 提交(基址: str, 载荷: dict, 超时秒: float = 320.0):
    """返回 (响应 dict, 错误文本)。HTTP 错误也会尝试读出响应体。"""
    请求 = urllib.request.Request(
        基址 + "/v1/cache/rebuild",
        data=json.dumps(载荷).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(请求, timeout=超时秒) as 响应:
            return json.loads(响应.read()), None
    except urllib.error.HTTPError as 错:
        try:
            体 = json.loads(错.read())
        except Exception:
            体 = {"raw": f"HTTP {错.code}"}
        return 体, f"HTTP {错.code}"
    except Exception as 错:
        return None, f"{type(错).__name__}: {错}"


# ------------------------------------------------------------------ 账本计算
def 单位字节(几何: dict) -> dict:
    单位 = 几何.get("unit_bytes") or {}
    return {
        "kv": 单位.get("kv_per_token") or 20480,
        "moe": 单位.get("moe_per_expert") or 1775616,
        "mamba": 单位.get("mamba_per_slot") or 64389120,
    }


def 估算总池(几何: dict, 单位: dict) -> int:
    """
    从当前几何反推共享池总量。

    对「恰好用满」的配置（出厂就是）精确；对没用满的配置只是下界，
    所以调用方必须依赖服务端校验兜底（rejected 时会回报真实 budget）。
    """
    return (单位["kv"] * 几何.get("num_pages", 0)
            + 单位["moe"] * 几何.get("moe_cache_size", 0)
            + 单位["mamba"] * 几何.get("num_mamba_slots", 0))


def 分量(kv页, moe专家, mamba槽, 单位) -> dict:
    return {
        "KV": {"bytes": 单位["kv"] * kv页, "量": f"{kv页} token"},
        "MoE": {"bytes": 单位["moe"] * moe专家, "量": f"{moe专家} 专家"},
        "Mamba": {"bytes": 单位["mamba"] * mamba槽, "量": f"{mamba槽} 槽"},
    }


def 打印账本(几何: dict, 单位: dict, 标题: str = "当前切分"):
    分 = 分量(几何["num_pages"], 几何["moe_cache_size"],
             几何["num_mamba_slots"], 单位)
    总 = sum(x["bytes"] for x in 分.values())
    print(f"  {标题}：")
    for 名, 值 in 分.items():
        print(f"    {名:<6} {值['量']:>12}   {值['bytes'] / GiB:6.3f} GiB "
              f" {值['bytes'] / 总 * 100:5.1f}%")
    print(f"    {'合计':<6} {'':>12}   {总 / GiB:6.3f} GiB")
    return 总


def 求最小内存(单位: dict, 总池: int, 目标KV: int, 保留MoE: int):
    """在满足 KV + MoE 的前提下，mamba 最多能留几槽。"""
    剩 = 总池 - 单位["kv"] * 目标KV - 单位["moe"] * 保留MoE
    if 剩 < 0:
        return None, 剩
    return int(剩 // 单位["mamba"]), 剩


# ------------------------------------------------------------------ 主流程
def main() -> int:
    解析器 = argparse.ArgumentParser(
        description="FreeToken 缓存几何调优：用闲置的 Mamba 并发槽换上下文长度",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    解析器.add_argument("--base", default=基址默认, help=f"服务地址（默认 {基址默认}）")
    解析器.add_argument("--context", type=int, default=None,
                        help="目标上下文 token 数（默认取 /v1/models 的 max_model_len）")
    解析器.add_argument("--moe", type=int, default=None, help="保留的 MoE 专家数（默认保持现值）")
    解析器.add_argument("--mamba", type=int, default=None, help="手工指定 Mamba 槽数")
    解析器.add_argument("--restore", action="store_true", help="恢复出厂几何")
    解析器.add_argument("--wait", action="store_true", help="服务忙时轮询等待，而不是直接退出")
    解析器.add_argument("--wait-秒", type=float, default=180.0, help="等待上限（默认 180s）")
    解析器.add_argument("--dry-run", action="store_true", help="只算不提交，看看会改成什么")
    参数 = 解析器.parse_args()

    状态, 错 = 取状态(参数.base)
    if 错:
        print(f"✗ 连不上 {参数.base} —— 服务没起？\n  {错}")
        return 2

    几何 = 状态.get("geometry") or {}
    单位 = 单位字节(几何)
    上限 = (几何.get("limits") or {})

    print("═" * 70)
    print(" FreeToken 缓存几何")
    print("═" * 70)
    print(f"  服务        {参数.base}")
    print(f"  状态        {状态.get('state')}   last_rebuild="
          f"{'无' if not 状态.get('last_rebuild') else 状态['last_rebuild'].get('status')}")
    声明上限, _ = 取模型(参数.base)
    print(f"  /v1/models 声明 max_model_len = {声明上限}")
    总池 = 打印账本(几何, 单位)
    print(f"    （共享池总量按当前几何反推 ≈ {总池 / GiB:.3f} GiB）")
    print()

    # ---------------- restore 分支
    if 参数.restore:
        return 执行重建(参数, {"num_pages": 出厂["num_pages"],
                              "moe_cache_size": 出厂["moe_cache_size"],
                              "num_mamba_slots": 出厂["num_mamba_slots"]},
                        几何, 单位, "恢复出厂几何")

    # ---------------- 目标上下文
    # 只读是默认行为：不显式给 --context / --restore 就绝不碰服务
    显式目标 = 参数.context is not None
    目标 = 参数.context or 声明上限 or 几何["num_pages"]
    保留MoE = 参数.moe if 参数.moe is not None else 几何["moe_cache_size"]

    需要KV = 单位["kv"] * 目标
    kvmoe = 需要KV + 单位["moe"] * 保留MoE
    print(f"  目标上下文  {目标} token  →  KV 需要 {需要KV / GiB:.3f} GiB")

    if 参数.mamba is not None:
        选定槽 = 参数.mamba
    else:
        选定槽, 剩 = 求最小内存(单位, 总池, 目标, 保留MoE)
        if 选定槽 is None:
            print(f"  ✗ 目标放不下：KV + MoE 就要 {kvmoe / GiB:.3f} GiB，"
                  f"已超共享池 {总池 / GiB:.3f} GiB")
            print(f"    要么把 --context 降到 {int((总池 - 单位['moe'] * 保留MoE) / 单位['kv'])} 以下，"
                  f"要么把 --moe 降到 {int((总池 - 需要KV) / 单位['moe'])} 以下")
            return 3

    槽下限 = (上限.get("mamba_slots") or {}).get("min", 16)
    槽上限 = (上限.get("mamba_slots") or {}).get("max", 58)
    选定槽 = max(槽下限, min(槽上限, 选定槽))

    预计 = 分量(目标, 保留MoE, 选定槽, 单位)
    预计总量 = sum(x["bytes"] for x in 预计.values())
    print(f"  建议切分    KV={目标}  MoE={保留MoE}  Mamba={选定槽}")
    print(f"    占用      {预计总量 / GiB:.3f} GiB，"
          f"余量 {(总池 - 预计总量) / MiB:.1f} MiB")
    print()
    print(f"  ⓘ 并发上限将从 {几何['num_mamba_slots']} 变成 {选定槽}"
          f"（单用户无碍；要跑 {选定槽}+ 路并发就别这么调）")
    if 选定槽 < 几何["num_mamba_slots"]:
        print(f"  ⓘ 上下文 {几何['num_pages']} → {目标} token"
              f"（+{(目标 - 几何['num_pages']) / 几何['num_pages'] * 100:.0f}%）")
    print()

    载荷 = {"num_pages": 目标, "moe_cache_size": 保留MoE,
            "num_mamba_slots": 选定槽}

    if not 显式目标 and not 参数.dry_run:
        print(f"  ⓘ 只读模式 —— 以上是现状与建议，**服务未被改动**。")
        print(f"    真要调：加 --context {目标}；想先看会改成什么：再加 --dry-run；")
        print(f"    想恢复出厂：--restore --dry-run。")
        return 0
    return 执行重建(参数, 载荷, 几何, 单位, "应用新切分")


def 等服务空闲(参数, 时长上限: float) -> bool:
    """轮询直到没有活动请求。返回是否成功等到了空闲。"""
    起点 = time.time()
    while time.time() - 起点 < 时长上限:
        统计, 错 = 取统计(参数.base)
        if 错:
            return False
        活动 = (统计.get("requests") or {}).get("active", 0)
        槽 = (统计.get("mamba") or {}).get("used_slots", 0)
        if 活动 == 0 and 槽 == 0:
            return True
        print(f"    · 服务忙（active={活动}, used_slots={槽}），等 3s …")
        time.sleep(3)
    return False


def 执行重建(参数, 载荷: dict, 几何: dict, 单位: dict, 标题: str) -> int:
    print(f"  ▶ {标题}")
    载荷 = dict(载荷)
    载荷.setdefault("mode", "if_idle")
    载荷.setdefault("timeout", 280)
    print(f"    请求体  {json.dumps(载荷, ensure_ascii=False)}")
    if 参数.dry_run:
        print("    ⓘ --dry-run：只算不提交，服务未做任何改动")
        return 0

    # 当服务端回报 rejected 时，用真实 budget 重算 mamba 槽后重试
    for 第次 in range(4):
        起点 = time.time()
        响应, 错 = 提交(参数.base, 载荷)
        耗时 = time.time() - 起点
        if 错 and not 响应:
            print(f"    ✗ 提交失败（{耗时:.1f}s）：{错}")
            return 4

        结果 = (响应 or {}).get("status")
        提示 = (响应 or {}).get("error") or ""

        if 结果 == "ok":
            print(f"    ✓ 成功（{耗时:.1f}s）"
                  f"  KV={响应.get('num_pages')}  MoE={响应.get('moe_cache_size')}"
                  f"  Mamba={响应.get('mamba_slots')}")
            print()
            print("  ⚠ 这是**运行时**调整，服务重启后会回到启动时的切分。")
            print("    要固化，把 serve_freetoken.sh 的 KV_TUNING 打开（见该脚本顶部）。")
            return 0

        if 结果 == "busy":
            print(f"    · 服务忙，未改动任何配置（{耗时:.1f}s）")
            if not 参数.wait:
                print("    重跑时加 --wait，或等服务闲下来再试。")
                return 5
            if not 等服务空闲(参数, 参数.wait_秒):
                print("    ✗ 等待超时，放弃")
                return 5
            continue

        if 结果 == "rejected":
            print(f"    · 被拒：{提示}")
            # 从提示里抠出真实 budget（形如 "... > budget 2.47 GiB"），据此重算
            预算 = None
            匹配 = re.search(r"budget\s+([\d.]+)\s*GiB", 提示)
            if 匹配:
                预算 = int(float(匹配.group(1)) * GiB)
                print(f"    · 服务端真实可用预算 {(预算 or 0) / GiB:.3f} GiB，按它重算")
            if 预算:
                可用 = 预算 - 单位["kv"] * 载荷["num_pages"]
                新槽 = int(可用 // 单位["mamba"])
                新槽 = max((上限下限(几何) or 16), 新槽)
                if 新槽 >= 载荷["num_mamba_slots"]:
                    print("    ✗ 重算后槽数没有减少，无法再降，放弃")
                    return 6
                print(f"    · Mamba {载荷['num_mamba_slots']} → {新槽} 重试")
                载荷["num_mamba_slots"] = 新槽
                continue
            print("    ✗ 原始配置保持不变（服务仍在正常服务）")
            return 6

        print(f"    ? 未知结果：{响应}")
        return 7

    print("    ✗ 重试次数用尽")
    return 7


def 上限下限(几何: dict):
    return ((几何.get("limits") or {}).get("mamba_slots") or {}).get("min")


if __name__ == "__main__":
    sys.exit(main())
