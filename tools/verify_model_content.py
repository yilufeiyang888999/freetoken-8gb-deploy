#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型权重内容级校验 —— 字节数对得上，不代表内容是对的。

为什么需要这个：
    下载器（迅雷、IDM 等）普遍会**预分配磁盘空间**，先把文件撑到目标大小，
    再往里填数据。所以"大小正确"只能证明任务建对了，不能证明下完了。
    断点续传、崩溃恢复、预分配之后中断，都可能留下
    「大小正确 + 内容是空洞/半截」的假完成文件。

三层校验（从便宜到贵）：
    1. 结构校验 —— safetensors 头部记录了每个 tensor 的 data_offsets，
       最大值 + 8 + header长度 必须严格等于文件大小。等式不成立 = 文件坏了。
       只读几 KB，秒出，能抓出截断和大部分预分配异常。
    2. 空洞采样 —— 在大文件里均匀取若干点，每点读 4KB 看是否全为 0x00。
       全部采样点都是零 = 高度疑似预分配空洞。真实权重不可能处处为零。
    3. 全量哈希 —— 加上 --sha256 时取官方哈希做全量比对，最强判据，
       但要读满 22GB，约 1~3 分钟。

       官方哈希从 /api/models/{repo}/tree/{rev}?recursive=1 取，
       分两种算法，17 个文件全都能核：
         LFS 大文件（5 个）  -> lfs.oid，就是内容 sha256
         其余小文件（12 个） -> 条目 oid，是 git blob 的 sha1，
                                本地按 sha1("blob <size>\\0" + 内容) 复算

用法：
    python verify_model_content.py D:\\models\\Qwen3.6-35B-A3B-NVFP4
    python verify_model_content.py <目录> --sha256
"""

import argparse
import hashlib
import json
import re
import struct
import sys
import urllib.request
from pathlib import Path

模型仓库 = "nvidia/Qwen3.6-35B-A3B-NVFP4"
模型版本 = "1355db6a052410cfd62085d94b58866fd0f2c3c5"

# 文件名 -> 确切字节数（取自仓库 API）
期望清单 = {
    "model-00001-of-00003.safetensors": 10006877608,
    "model-00002-of-00003.safetensors": 10003595752,
    "model-00003-of-00003.safetensors": 3413864960,
    "model.safetensors.index.json": 13726227,
    "tokenizer.json": 12807982,
    "vocab.json": 6722759,
    ".quant_summary.txt": 4752286,
    "config.json": 58110,
    "hf_quant_config.json": 35085,
    "tokenizer_config.json": 16718,
    "README.md": 9936,
    "chat_template.jinja": 7764,
    ".gitattributes": 1635,
    "preprocessor_config.json": 390,
    "video_preprocessor_config.json": 385,
    "generation_config.json": 202,
    "configuration.json": 58,
}

大体量后缀 = (".safetensors",)
采样点数 = 12          # 每个大文件取多少个采样点
采样块大小 = 4096      # 每个采样点读多少字节


def 人类可读(字节: int) -> str:
    if 字节 >= 1024 ** 3:
        return f"{字节 / 1024 ** 3:.2f} GB"
    if 字节 >= 1024 ** 2:
        return f"{字节 / 1024 ** 2:.1f} MB"
    if 字节 >= 1024:
        return f"{字节 / 1024:.1f} KB"
    return f"{字节} B"


def 校验结构(路径: Path):
    """safetensors 结构校验。返回 (是否通过, 说明)。"""
    try:
        文件大小 = 路径.stat().st_size
        with 路径.open("rb") as f:
            头8 = f.read(8)
            if len(头8) != 8:
                return False, "文件不足 8 字节，连头部长度都读不到"
            头长 = struct.unpack("<Q", 头8)[0]

            # 头长是个小端 u64，一个合法 safetensors 的头长不可能这么大
            if 头长 == 0:
                return False, "头部长度为 0 —— 典型预分配空洞特征"
            if 头长 > min(100 * 1024 * 1024, 文件大小):
                return False, f"头部长度异常（{头长} 字节），超过文件本身或 100MB 上限"

            头Json = f.read(头长)
            if len(头Json) != 头长:
                return False, f"头部 JSON 被截断：实读 {len(头Json)} / 声明 {头长}"

        try:
            header = json.loads(头Json.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            return False, f"头部 JSON 解析失败：{e}"

        元数据 = header.pop("__metadata__", None)
        if not header:
            return False, "头部里没有任何 tensor 条目"

        # 关键等式：数据段起点 + 最大偏移 == 文件大小
        最大偏移 = 0
        tensor数 = 0
        最大张量 = ""
        for 名, 信息 in header.items():
            偏移 = 信息.get("data_offsets")
            if not (isinstance(偏移, list) and len(偏移) == 2):
                return False, f"tensor {名} 的 data_offsets 字段异常"
            if 偏移[1] > 最大偏移:
                最大偏移 = 偏移[1]
                最大张量 = 名
            tensor数 += 1

        期望文件大小 = 8 + 头长 + 最大偏移
        if 期望文件大小 != 文件大小:
            return False, (f"长度对不上：header 声明应为 {期望文件大小} 字节"
                           f"（8 头长 + {头长} 头 + {最大偏移} 数据），"
                           f"实际 {文件大小}，差 {文件大小 - 期望文件大小:+d}")

        # 采样空洞检测
        采样 = []
        最多可读 = max(0, 文件大小 - 采样块大小)
        本次块大小 = min(采样块大小, 文件大小)   # 文件比采样块还小时就整块读
        with 路径.open("rb") as f:
            for i in range(采样点数):
                # 从数据段里均匀取样，避开头部。
                # 必须夹在 [0, 文件大小 - 采样块] 之间：算出来是负位置时，
                # f.seek(负数) 会直接抛 OSError。
                位置 = 8 + 头长 + (最大偏移 * i // 采样点数)
                位置 = max(0, min(位置, 最多可读))
                f.seek(位置)
                块 = f.read(本次块大小)
                采样.append(块)

        全零块数 = sum(1 for 块 in 采样 if 块 and not any(块))
        if 全零块数 == len(采样):
            return False, f"{len(采样)} 个采样点全是 0x00 —— 几乎是预分配空洞，没下到真数据"
        if 全零块数 > len(采样) // 2:
            return False, f"{全零块数}/{len(采样)} 个采样点是 0x00 —— 可疑，数据可能有大片空洞"

        说明 = (f"{tensor数} 个 tensor，最大张量 {最大张量}，"
                f"结构自洽（8 + {头长} + {最大偏移} = {文件大小}），"
                f"采样 {len(采样)} 点无空洞")
        return True, 说明

    except OSError as e:
        return False, f"读取失败：{e}"


def 校验Json(路径: Path):
    """JSON 文件解析校验。"""
    try:
        with 路径.open("rb") as f:
            原始 = f.read()
        if not 原始.strip():
            return False, "空文件"
        数据 = json.loads(原始.decode("utf-8"))
        条目 = len(数据) if isinstance(数据, (dict, list)) else 1
        return True, f"合法 JSON（{条目} 个 {'键' if isinstance(数据, dict) else '元素'}）"
    except UnicodeDecodeError as e:
        return False, f"不是合法 UTF-8：{e}"
    except json.JSONDecodeError as e:
        return False, f"JSON 解析失败：{e}"
    except OSError as e:
        return False, f"读取失败：{e}"


def 校验文本(路径: Path):
    """文本文件基本校验：能解码、不是全空白、不是错误页。"""
    try:
        with 路径.open("rb") as f:
            原始 = f.read()
        if not 原始.strip():
            return False, "空文件"
        文本 = 原始.decode("utf-8", errors="strict")
        # 常见的下载失败特征：拿到的是 HTML 错误页
        if re.match(r"^\s*<!DOCTYPE html|^\s*<html", 文本, re.I):
            return False, "内容是一个 HTML 页面 —— 多半下到了错误页，不是真文件"
        return True, f"合法 UTF-8 文本（{len(文本)} 字符）"
    except UnicodeDecodeError as e:
        return False, f"不是合法 UTF-8：{e}"
    except OSError as e:
        return False, f"读取失败：{e}"


def 取官方哈希(超时=25):
    """向仓库 API 取官方哈希，返回 {文件名: (算法, 哈希)}，取不到返回空字典。

    端点必须用 /tree/ 而不是 /api/models/{repo}：
        - /api/models/{repo} 的 siblings **不返回 lfs 字段**（实测 17 条全是空）
        - /tree/{rev}?recursive=1 才带 lfs.oid（即文件内容 sha256）
        - 别用 resolve 直链的 ETag：那是 40 位 git blob sha1（且带 W/ 弱校验前缀），
          不是 sha256，拿来比对必然全错

    非 LFS 的小文件没有 lfs 字段，但有条目级 oid —— 那是 git blob 的 sha1，
    可以本地按 `sha1("blob <size>\\0" + 内容)` 复算，所以 17 个文件全都能核。
    """
    结果 = {}
    for 主机 in ("hf-mirror.com", "huggingface.co"):
        url = f"https://{主机}/api/models/{模型仓库}/tree/{模型版本}?recursive=1"
        try:
            请求 = urllib.request.Request(url, headers={"User-Agent": "verify-model-content/1.0"})
            with urllib.request.urlopen(请求, timeout=超时) as 响应:
                数据 = json.loads(响应.read().decode("utf-8"))
        except Exception as e:
            print(f"  取哈希失败（{主机}）：{type(e).__name__}: {e}")
            continue

        条目列表 = 数据 if isinstance(数据, list) else 数据.get("tree", [])
        for 条目 in 条目列表:
            if not isinstance(条目, dict):
                continue
            名 = 条目.get("path", "")
            if not 名:
                continue
            lfs = 条目.get("lfs") or {}
            if len(lfs.get("oid", "")) == 64:
                结果[名] = ("sha256", lfs["oid"].lower())
            elif len(条目.get("oid", "")) == 40:
                结果[名] = ("git-sha1", 条目["oid"].lower())

        if 结果:
            print(f"  已从 {主机} 取到 {len(结果)} 个官方哈希")
            return 结果
    return 结果


def 算哈希(路径: Path, 算法: str, 分块=8 * 1024 * 1024):
    """按指定算法算文件哈希。

    git-sha1 不是直接对内容算 sha1，而是 git 的对象哈希：
        sha1(b"blob " + str(大小) + b"\\0" + 内容)
    少算了那个前缀，结果一定对不上。
    """
    总大小 = 路径.stat().st_size
    if 算法 == "sha256":
        散列 = hashlib.sha256()
    elif 算法 == "git-sha1":
        散列 = hashlib.sha1()
        散列.update(f"blob {总大小}\0".encode())
    else:
        raise ValueError(f"未知算法：{算法}")

    已读 = 0
    with 路径.open("rb") as f:
        while True:
            块 = f.read(分块)
            if not 块:
                break
            散列.update(块)
            已读 += len(块)
            if 总大小 > 1024 ** 3:      # 大文件才打进度，小文件秒完
                print(f"\r     {已读 * 100 // 总大小:3d}%", end="", flush=True)
    if 总大小 > 1024 ** 3:
        print("\r     100%", flush=True)
    return 散列.hexdigest()


def 主流程():
    parser = argparse.ArgumentParser(description="模型权重内容级校验")
    parser.add_argument("模型目录", help="模型文件所在目录")
    parser.add_argument("--sha256", action="store_true",
                        help="额外做全量 sha256 比对（要读满 22GB，约 1~3 分钟）")
    args = parser.parse_args()

    目录 = Path(args.模型目录).expanduser()
    if not 目录.is_dir():
        print(f"!! 目录不存在：{目录}")
        return 2

    print("=" * 78)
    print(" 模型权重内容级校验")
    print("=" * 78)
    print(f" 目录：{目录}")
    print()

    # ------------------------------------------------ 第 1 层：存在 + 字节数
    print("-" * 78)
    print(" 第 1 层　文件存在性与字节数")
    print("-" * 78)

    缺失 = []
    大小错 = []
    for 名, 期望 in 期望清单.items():
        路径 = 目录 / 名
        if not 路径.is_file():
            缺失.append(名)
            print(f"  [缺失] {名}")
            continue
        实际 = 路径.stat().st_size
        if 实际 != 期望:
            大小错.append(名)
            print(f"  [字节数错] {名}  实际 {实际} / 期望 {期望}"
                  f"（差 {实际 - 期望:+d}）")
    if 缺失 or 大小错:
        print(f"\n  第 1 层就不过：缺 {len(缺失)} 个，字节数错 {len(大小错)} 个")
        return 1
    print(f"  [通过] 17 个文件全部存在且字节数完全一致")

    # ------------------------------------------------ 第 2 层：内容
    print()
    print("-" * 78)
    print(" 第 2 层　内容校验（结构 / 空洞采样 / JSON 合法性）")
    print("-" * 78)

    内容失败 = []
    内容明细 = {}   # 名字 -> 判断类型，供 --sha256 复用

    for 名 in 期望清单:
        路径 = 目录 / 名
        if 名.endswith(大体量后缀):
            print(f"  --> {名}（{人类可读(路径.stat().st_size)}）")
            通过, 说明 = 校验结构(路径)
        elif 名.endswith(".json"):
            通过, 说明 = 校验Json(路径)
        else:
            通过, 说明 = 校验文本(路径)

        if 通过:
            if not 名.endswith(大体量后缀):
                print(f"  [通过] {名} —— {说明}")
            else:
                print(f"         {说明}")
        else:
            内容失败.append((名, 说明))
            print(f"  [失败] {名} —— {说明}")

    if 内容失败:
        print(f"\n  第 2 层不过，{len(内容失败)} 个文件内容有问题：")
        for 名, 说明 in 内容失败:
            print(f"    - {名}：{说明}")
        print("\n  这些都是「大小对但内容坏」的典型症状。别拿去启动服务，")
        print("  把目录清空重新下载（或让下载器重新校验任务）。")
        return 1
    print(f"\n  [通过] 17 个文件内容结构全部正常")

    # ------------------------------------------------ 第 3 层：sha256
    if not args.sha256:
        print()
        print("=" * 78)
        print(" 结论：结构级校验全部通过")
        print("=" * 78)
        print("  想再加一层保险（防止数据段内部有静默损坏），加 --sha256 重跑。")
        return 0

    print()
    print("-" * 78)
    print(" 第 3 层　全量 sha256 比对")
    print("-" * 78)

    官方哈希 = 取官方哈希()
    if not 官方哈希:
        print("  [跳过] 一个官方哈希都没取到，第 3 层未执行")
        print("         注意：这是「没能验证」，不是「验证失败」——")
        print("         前两层（字节数 + 内容结构）已经通过，文件可以用。")
        print("         可自查：curl -sI https://hf-mirror.com/api/models/"
              f"{模型仓库} | head -3")
        return 0

    哈希失败 = []
    无对照 = []
    通过数 = 0
    for 名 in 期望清单:
        条目 = 官方哈希.get(名)
        if not 条目:
            无对照.append(名)
            print(f"  [跳过] {名} —— 仓库未提供哈希")
            continue
        算法, 期望哈希 = 条目
        路径 = 目录 / 名
        print(f"  --> {名}（{算法}）")
        实际哈希 = 算哈希(路径, 算法)
        if 实际哈希 == 期望哈希:
            通过数 += 1
            print(f"  [通过] {名}")
        else:
            哈希失败.append(名)
            print(f"  [失败] {名}")
            print(f"         期望 {期望哈希}")
            print(f"         实际 {实际哈希}")

    print()
    print("=" * 78)
    if 哈希失败:
        print(f" 结论：哈希有 {len(哈希失败)} 个对不上 —— 文件已损坏，必须重下")
        for 名 in 哈希失败:
            print(f"    - {名}")
        print("=" * 78)
        return 1
    print(f" 结论：全部通过（哈希比对 {通过数} 个"
          f"{f'，{len(无对照)} 个无对照跳过' if 无对照 else ''}）")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(主流程())
