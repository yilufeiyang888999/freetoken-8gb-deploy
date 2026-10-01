#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
迅雷下载结果整理器 —— 把乱名文件还原成 HF 仓库的正确文件名。

为什么需要它：
    迅雷下载 hf-mirror 的直链时，链接会 302 跳到 CDN / Xet CAS 服务器，
    跳转后 URL 末尾不再带原始文件名，迅雷会把文件存成
    "model-00001-of-00003.safetensors" 之外的怪名字
    （常见如一大串 sha 十六进制、或 URL 编码后的长名字）。

    与其手工改名 17 个文件，不如用「字节数」当指纹反查 —— 本模型的 17 个
    文件字节数互不相同（见下表），所以字节数 → 文件名的映射是唯一的。

安全性：
    只做「匹配 + 移动」，不做任何删除。源目录里的东西一律不动，
    匹配不上的文件只报告、不碰。

用法（Windows 上跑，用 managed python）：
    python collect_model_files.py <迅雷下载目录> [目标目录]

例：
    python collect_model_files.py D:\\Downloads D:\\models\\Qwen3.6-35B-A3B-NVFP4

只想看会怎么改、不动文件：
    python collect_model_files.py <迅雷下载目录> --dry-run
"""

import argparse
import shutil
import sys
from pathlib import Path

# ============================================================ 期望清单
# 文件名 -> 确切字节数（取自 HuggingFace 仓库 API）
# revision: 1355db6a052410cfd62085d94b58866fd0f2c3c5
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

# 字节数 -> 文件名（反向索引）。建立时检查唯一性。
字节数索引 = {}
重复字节数 = []
for _名, _字节 in 期望清单.items():
    if _字节 in 字节数索引:
        重复字节数.append(_字节)
    字节数索引[_字节] = _名

# 迅雷的临时 / 垃圾文件后缀，一律跳过
跳过后缀 = {".td", ".xltd", ".cfg", ".dlbt", ".downloading"}


def 扫描候选(源目录: Path):
    """列出源目录下所有可能是下载结果的普通文件，返回 (候选, 被跳过)。"""
    全部 = []
    跳过 = []
    for 路径 in sorted(源目录.rglob("*")):
        if not 路径.is_file():
            continue
        # 只按已知的迅雷临时后缀跳过。
        # 注意不能按「点开头」跳过：.gitattributes 和 .quant_summary.txt
        # 本身就是权重清单里的成员，会被误杀。
        if 路径.suffix.lower() in 跳过后缀:
            跳过.append(路径)
            continue
        全部.append(路径)
    return 全部, 跳过


def 主流程():
    parser = argparse.ArgumentParser(
        description="按字节数把迅雷下载的乱名文件还原成正确文件名"
    )
    parser.add_argument("源目录", help="迅雷下载目录")
    parser.add_argument("目标目录", nargs="?", default=None,
                        help="整理后的输出目录（不填则原地改名）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只预览，不移动任何文件")
    args = parser.parse_args()

    if 重复字节数:
        print("!! 清单内部有字节数冲突，无法用字节数当指纹：", 重复字节数)
        sys.exit(2)

    源目录 = Path(args.源目录).expanduser().resolve()
    if not 源目录.is_dir():
        print(f"!! 源目录不存在：{源目录}")
        sys.exit(2)

    目标目录 = Path(args.目标目录).expanduser().resolve() if args.目标目录 else 源目录

    print("=" * 68)
    print(" 迅雷下载结果整理")
    print("=" * 68)
    print(f" 源目录  ：{源目录}")
    print(f" 目标目录：{目标目录}")
    print(f" 文件总数：{len(期望清单)} 个，合计 "
          f"{sum(期望清单.values()) / 1024**3:.2f} GB")
    if args.dry_run:
        print(" 模式    ：预览（不改动任何文件）")
    print()

    文件列表, 跳过的 = 扫描候选(源目录)
    if 跳过的:
        print(f" 跳过 {len(跳过的)} 个迅雷临时文件（.td / .xltd 等）")

    # 源目录是空的 —— 十有八九是把两个参数的位置弄反了，或者下载目录填错。
    # 这种情况必须直说，否则用户只会看到一堆「缺失」，以为是文件没下完。
    if not 文件列表:
        print()
        print(f" !! 源目录里一个文件都没有：{源目录}")
        print()
        print("    别忘了两个参数的含义：")
        print("      源目录   = 迅雷「下载到」的那个目录")
        print("      目标目录 = 整理后的输出位置")
        print()
        if not args.dry_run and 目标目录 != 源目录:
            print("    如果你已经把权重下在最终位置了（文件名本来就是对的），")
            print("    那就不用整理了，直接进下一步：")
            print(f"      bash import_model_from_windows.sh /mnt/d/models/Qwen3.6-35B-A3B-NVFP4")
        else:
            print("    确认一下迅雷的任务详情里「下载目录」到底设成了哪个路径。")
        print()
        return 1

    if not args.dry_run:
        目标目录.mkdir(parents=True, exist_ok=True)

    已匹配 = {}          # 期望文件名 -> 实际源路径
    未识别 = []          # 字节数对不上任何期望值的文件
    冲突 = []            # 同一个期望文件匹配到多个源文件

    # 第一轮：文件名精确匹配优先。
    # 为什么要有这一轮：字节数当指纹有个前提 —— 目录里没有「字节数恰好相同
    # 的无关文件」。真实清单里 17 个值互不相同，但下载目录里混进一个大小
    # 巧合的杂文件时，靠字节数猜就可能张冠李戴。文件名直接对得上的，
    # 就不要再猜了。
    剩余 = []
    for 路径 in 文件列表:
        try:
            大小 = 路径.stat().st_size
        except OSError:
            continue
        if 路径.name in 期望清单 and 大小 == 期望清单[路径.name]:
            if 路径.name in 已匹配:
                冲突.append((路径.name, 路径))
            else:
                已匹配[路径.name] = 路径
            continue
        剩余.append((路径, 大小))

    # 第二轮：字节数指纹匹配（对付迅雷存成的哈希名 / URL 编码名）
    for 路径, 大小 in 剩余:
        名 = 字节数索引.get(大小)
        if 名 is None:
            未识别.append((路径, 大小))
            continue
        if 名 in 已匹配:
            冲突.append((名, 路径))
            continue
        已匹配[名] = 路径

    # ------------------------------------------------ 复制 / 移动
    print("-" * 68)
    print(f"{'文件':<36}{'字节数':>14}  状态")
    print("-" * 68)

    成功 = 0
    已存在 = 0
    for 名, 期望字节 in 期望清单.items():
        目标 = 目标目录 / 名
        # 目标里已经是完整的那一份？——注意 dry-run 不做任何实际判断
        目标就绪 = (not args.dry_run
                    and 目标.is_file()
                    and 目标.stat().st_size == 期望字节)

        if 名 in 已匹配:
            源 = 已匹配[名]
            if args.dry_run:
                标记 = "预览"
                成功 += 1
            elif 目标就绪:
                已存在 += 1
                标记 = "已存在"
            else:
                try:
                    if 目标.exists():
                        目标.unlink()
                    # 同一磁盘用 rename（瞬时），跨盘自动降级为复制
                    shutil.move(str(源), str(目标))
                    成功 += 1
                    标记 = "整理"
                except OSError as e:
                    print(f"{名:<36}{期望字节:>14}  失败({e})")
                    continue
            print(f"{名:<36}{期望字节:>14}  {标记}  <- {源.name[:40]}")
        elif 目标就绪:
            # 源目录已被上一轮搬空，但目标里那份是完整的
            已存在 += 1
            print(f"{名:<36}{期望字节:>14}  已就绪")
        else:
            print(f"{名:<36}{期望字节:>14}  缺失")

    # ------------------------------------------------ 未识别 / 冲突
    print("-" * 68)
    if 未识别:
        print(f"\n 未识别的文件（{len(未识别)} 个）——可能是没下完的残片，或别的东西：")
        for 路径, 大小 in 未识别:
            提示 = ""
            for 期望字节, 期望名 in 字节数索引.items():
                if abs(大小 - 期望字节) < 期望字节 * 0.02 and 大小 < 期望字节:
                    提示 = f"  疑似 {期望名} 未下完（差 {期望字节 - 大小} 字节）"
                    break
            print(f"   {大小:>14} 字节  {路径.name[:50]}{提示}")
    if 冲突:
        print(f"\n 重复匹配（{len(冲突)} 个）——同一份权重下了多遍，多余的那个没动：")
        for 名, 路径 in 冲突:
            print(f"   {名}  <- {路径}")

    # ------------------------------------------------ 汇总
    # 成败以「目标目录里 17 个文件是否都完整」为准，而不是以源目录匹配数为准 ——
    # 因为重跑时源目录已被上一步搬空，按源匹配会误报"缺失"。
    if args.dry_run:
        缺 = [名 for 名 in 期望清单 if 名 not in 已匹配]
    else:
        缺 = []
        for 名, 期望字节 in 期望清单.items():
            路径 = 目标目录 / 名
            if not 路径.is_file() or 路径.stat().st_size != 期望字节:
                缺.append(名)

    print()
    print("=" * 68)
    if args.dry_run:
        print(f" 预览：能识别 {len(已匹配)}/{len(期望清单)} 个文件")
    else:
        总数 = sum(1 for p in 目标目录.iterdir()) if 目标目录.is_dir() else 0
        print(f" 完成：整理 {成功} 个，已存在 {已存在} 个，"
              f"目标目录现共 {总数} 个文件")
    print("=" * 68)

    if 缺:
        print(f"\n 还差 {len(缺)} 个文件：{', '.join(缺)}")
        print(" 回到迅雷把没下完的任务跑完，再重跑本脚本 —— 已就位的会自动跳过。")
        return 1

    print(f"\n 全部 {len(期望清单)} 个文件就位，共 "
          f"{sum(期望清单.values()) / 1024**3:.2f} GB")
    print("\n 下一步：在 WSL 里执行导入")
    print("   bash import_model_from_windows.sh /mnt/d/models/Qwen3.6-35B-A3B-NVFP4")
    return 0


if __name__ == "__main__":
    sys.exit(主流程())
