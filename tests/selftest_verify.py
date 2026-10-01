# -*- coding: utf-8 -*-
"""verify_model_content.py 的判别力自测。

一个永远返回「通过」的校验脚本等于安慰剂。这个测试专门造坏文件，
验证脚本能不能把它们**拒掉** —— 拒不掉就是假的安心。

覆盖：
    好文件          -> 必须通过
    头部长度 = 0    -> 必须拒（典型预分配空洞）
    数据段截断      -> 必须拒（结构等式不成立）
    数据段全零      -> 必须拒（空洞采样）
    头部 JSON 损坏  -> 必须拒
    HTML 错误页     -> 必须拒（文本校验）
    非法 JSON       -> 必须拒（JSON 校验）
    小文件不崩      -> seek 负位置不能抛异常
"""
import importlib.util
import json
import shutil
import struct
import sys
import tempfile
from pathlib import Path

脚本 = Path(__file__).resolve().parent.parent / "tools" / "verify_model_content.py"
spec = importlib.util.spec_from_file_location("vmc", 脚本)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

临时根 = Path(tempfile.mkdtemp(prefix="vmc_selftest_"))


def 造safetensors(路径: Path, 数据字节: bytes = b"\x01" * 16,
                  篡改=None):
    """构造一个最小可用的 safetensors 文件。篡改=callable(头长, 头Json, 数据)->bytes"""
    头 = {
        "weight": {"dtype": "F32", "shape": [4], "data_offsets": [0, len(数据字节)]},
        "__metadata__": {"format": "pt"},
    }
    头Json = json.dumps(头, separators=(",", ":")).encode()
    头长 = len(头Json)
    原始 = struct.pack("<Q", 头长) + 头Json + 数据字节
    if 篡改:
        原始 = 篡改(头长, 头Json, 数据字节)
    路径.write_bytes(原始)
    return 路径


结果 = []


def 记(名, 通过, 附注=""):
    结果.append((名, 通过))
    print(f"  [{'通过' if 通过 else '失败'}] {名}{('　' + 附注) if 附注 else ''}")


print("=" * 70)
print(" 校验脚本判别力自测")
print("=" * 70)

# ---------------------------------------------------------- 1 好文件
print("\n--- 1. 正常文件应当通过 ---")
好 = 造safetensors(临时根 / "good.safetensors", b"\x01" * 16)
通过, 说明 = mod.校验结构(好)
记("好文件通过", 通过, 说明[:60])

# ---------------------------------------------------------- 2 头部长度=0
print("\n--- 2. 头部长度 0（预分配空洞）应当拒绝 ---")
零 = 造safetensors(临时根 / "zero.safetensors", b"\x01" * 16,
                   篡改=lambda 头长, 头Json, 数据: struct.pack("<Q", 0) + 头Json + 数据)
通过2, 说明2 = mod.校验结构(零)
记("拒掉头部长度 0", not 通过2, 说明2[:60])

# ---------------------------------------------------------- 3 数据段截断
print("\n--- 3. 数据段截断应当拒绝 ---")
截 = 造safetensors(临时根 / "trunc.safetensors", b"\x01" * 16,
                   篡改=lambda 头长, 头Json, 数据: struct.pack("<Q", 头长) + 头Json + 数据[:8])
通过3, 说明3 = mod.校验结构(截)
记("拒掉截断文件", not 通过3, 说明3[:70])

# ---------------------------------------------------------- 4 数据段全零
print("\n--- 4. 数据段全零（空洞）应当拒绝 ---")
洞 = 造safetensors(临时根 / "hole.safetensors", b"\x00" * 4096)
通过4, 说明4 = mod.校验结构(洞)
记("拒掉全零数据段", not 通过4, 说明4[:70])

# ---------------------------------------------------------- 5 头部 JSON 损坏
print("\n--- 5. 头部 JSON 损坏应当拒绝 ---")
def 弄坏头Json(头长, 头Json, 数据):
    # 长度必须保持不变，才能确保被拒是因为 JSON 解析失败，
    # 而不是先被长度检查拦下（那样测的就是别的东西了）。
    # 用 '#' 填充：合法 JSON 里不可能出现裸的 # 字符。
    坏 = b"{" + b"#" * (头长 - 1)
    assert len(坏) == 头长 and 坏 != 头Json, "篡改无效 —— 测试本身写错了"
    return struct.pack("<Q", 头长) + 坏 + 数据

坏Json = 造safetensors(临时根 / "badjson.safetensors", b"\x01" * 16, 篡改=弄坏头Json)
通过5, 说明5 = mod.校验结构(坏Json)
记("拒掉头部 JSON 损坏", not 通过5, 说明5[:60])

# ---------------------------------------------------------- 6 HTML 错误页
print("\n--- 6. HTML 错误页应当拒绝 ---")
页 = 临时根 / "err.md"
页.write_bytes(b"<!DOCTYPE html>\n<html><body>404 Not Found</body></html>")
通过6, 说明6 = mod.校验文本(页)
记("拒掉 HTML 错误页", not 通过6, 说明6[:60])

# ---------------------------------------------------------- 7 非法 JSON
print("\n--- 7. 非法 JSON 应当拒绝 ---")
坏J = 临时根 / "bad.json"
坏J.write_bytes(b'{"a": 1,}')
通过7, 说明7 = mod.校验Json(坏J)
记("拒掉非法 JSON", not 通过7, 说明7[:60])

# ---------------------------------------------------------- 8 小文件不崩
print("\n--- 8. 极小文件不能因为 seek 负位置而崩 ---")
极小 = 造safetensors(临时根 / "tiny.safetensors", b"\x07" * 4)
try:
    通过8, 说明8 = mod.校验结构(极小)
    记("极小文件未抛异常", True, f"通过={通过8}")
except Exception as e:
    记("极小文件未抛异常", False, f"{type(e).__name__}: {e}")

# ---------------------------------------------------------- 9 真实数据（如果存在）
print("\n--- 9. 真实权重（存在就顺带核一下） ---")
真实 = Path("D:/models/Qwen3.6-35B-A3B-NVFP4/model-00003-of-00003.safetensors")
if 真实.is_file():
    通过9, 说明9 = mod.校验结构(真实)
    记("真实分片结构自洽", 通过9, 说明9[:80])
else:
    print("  [跳过] 真实权重不在本机，跳过")

# ---------------------------------------------------------- 汇总
失败 = sum(1 for _, p in 结果 if not p)
print("\n" + "=" * 70)
print(f" 结果：{len(结果) - 失败}/{len(结果)} 通过")
print("=" * 70)
if 失败:
    print(" !! 有判据失效 —— 这个校验脚本不能信，必须先修再用来下结论")

shutil.rmtree(临时根, ignore_errors=True)
sys.exit(1 if 失败 else 0)
