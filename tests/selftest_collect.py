# -*- coding: utf-8 -*-
"""collect_model_files.py 的逻辑自测：替换成小文件清单，验证
   ①哈希名按字节数还原 ②正确名不被乱改 ③精确名优先于字节数猜测
   ④临时文件/残片不被动 ⑤幂等"""
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

脚本 = Path(__file__).resolve().parent.parent / "tools" / "collect_model_files.py"
spec = importlib.util.spec_from_file_location("cmf", 脚本)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# 小清单：只保留 3 项，避免造 21GB 假文件。
# tokenizer.json 和另一个乱名文件同为 900 字节，用来验证「精确文件名优先」。
小清单 = {
    "tokenizer.json": 900,
    "config.json": 581,
    "vocab.json": 6722,
}
mod.期望清单 = 小清单
mod.字节数索引 = {v: k for k, v in 小清单.items()}

临时根 = Path(tempfile.mkdtemp(prefix="cmf_selftest_"))
源 = 临时根 / "downloads"
目标 = 临时根 / "organized"
源.mkdir()

(源 / "deadbeefcafe.bin").write_bytes(b"V" * 6722)      # 乱名 -> 应按字节数还原为 vocab.json
(源 / "config.json").write_bytes(b"C" * 581)             # 名字已对
(源 / "tokenizer.json").write_bytes(b"B" * 900)          # 名字已对，且与下面撞字节数
(源 / "hashname0.bin").write_bytes(b"A" * 900)           # 乱名 + 撞车 -> 应进冲突，不得覆盖
(源 / "zz900.bin.td").write_bytes(b"T" * 900)            # 迅雷临时文件 -> 跳过
(源 / "a3f9c2e1b4d5.safetensors").write_bytes(b"X" * 5000)  # 不在清单 -> 未识别，不碰
(源 / "9f8e7d.tokenizer.json").write_bytes(b"Z" * 450)   # 下了一半 -> 未识别，不碰

print("\n########## 第 1 轮：dry-run ##########")
sys.argv = ["collect_model_files.py", str(源), str(目标), "--dry-run"]
rc1 = mod.主流程()

断言 = [
    ("dry-run 返回 0（3/3 全识别）", rc1 == 0),
    ("dry-run 未创建目标目录", not 目标.exists()),
]

print("\n########## 第 2 轮：实际整理 ##########")
sys.argv = ["collect_model_files.py", str(源), str(目标)]
rc2 = mod.主流程()

print("\n########## 第 3 轮：重复执行（幂等） ##########")
sys.argv = ["collect_model_files.py", str(源), str(目标)]
rc3 = mod.主流程()

目标列表 = sorted(p.name for p in 目标.iterdir()) if 目标.is_dir() else []

断言 += [
    ("目标目录已创建", 目标.is_dir()),
    ("哈希名按字节数还原为 vocab.json", (目标 / "vocab.json").is_file()),
    ("vocab.json 内容与源一致", (目标 / "vocab.json").read_bytes() == b"V" * 6722),
    ("config.json 已就位", (目标 / "config.json").read_bytes() == b"C" * 581),
    ("tokenizer.json 已就位", (目标 / "tokenizer.json").is_file()),
    # 核心：900 字节撞车时，必须取名字正确的那个，不能被哈希名文件顶掉
    ("精确文件名优先于字节数猜测", (目标 / "tokenizer.json").read_bytes() == b"B" * 900),
    ("撞车的乱名文件未被搬走", (源 / "hashname0.bin").is_file()),
    ("目标目录只有 3 个文件", len(目标列表) == 3),
    ("不在清单的文件未进目标目录", "a3f9c2e1b4d5.safetensors" not in 目标列表),
    ("迅雷临时 .td 文件未被动", (源 / "zz900.bin.td").is_file()),
    ("下了一半的残片未被动", (源 / "9f8e7d.tokenizer.json").is_file()),
    ("残片内容未被覆盖", (源 / "9f8e7d.tokenizer.json").read_bytes() == b"Z" * 450),
    ("全部识别 -> 返回 0", rc2 == 0),
    ("幂等：第三轮仍返回 0", rc3 == 0),
    ("幂等：第三轮目标文件清单不变", sorted(p.name for p in 目标.iterdir()) == 目标列表),
]

失败 = 0
for 名, 结果 in 断言:
    print(f"  [{'通过' if 结果 else '失败'}] {名}")
    if not 结果:
        失败 += 1

print(f"\n 结果：{len(断言) - 失败}/{len(断言)} 通过")
print(" 目标目录：")
for p in sorted(目标.iterdir()):
    print(f"   {p.stat().st_size:>8} 字节  {p.name}")
print(" 源目录剩余：")
for p in sorted(源.iterdir()):
    print(f"   {p.stat().st_size:>8} 字节  {p.name}")

shutil.rmtree(临时根, ignore_errors=True)
sys.exit(1 if 失败 else 0)
