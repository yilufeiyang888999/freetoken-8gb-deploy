# -*- coding: utf-8 -*-
"""import_model_from_windows.sh 的沙箱测试。
把小文件清单注入脚本副本后，在真 bash 里跑完整流程。"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

工作区 = Path(__file__).resolve().parent.parent
原脚本 = (工作区 / "scripts" / "import_model_from_windows.sh").read_text(encoding="utf-8")

小清单 = [("a.bin", 1000), ("b.bin", 2000), ("c.bin", 3000)]
新FILES = "FILES=(\n" + "".join(f'  "{n}:{b}"\n' for n, b in 小清单) + ")"
测试内容 = re.sub(r"FILES=\(\n(?:.*\n)*?\)", 新FILES, 原脚本, count=1)
assert "a.bin:1000" in 测试内容, "FILES 数组替换失败"
assert "model-00001" not in 测试内容, "旧清单残留"


def 转bash(p) -> str:
    s = str(p).replace("\\", "/")
    if len(s) > 1 and s[1] == ":":
        s = "/" + s[0].lower() + s[2:]
    return s


临时根 = Path(tempfile.mkdtemp(prefix="ftimp_"))
测试脚本 = 临时根 / "import_test.sh"
测试脚本.write_text(测试内容, encoding="utf-8", newline="\n")

源 = 临时根 / "src"
目标 = 临时根 / "dst"
源.mkdir()


def 造源():
    if 源.exists():
        shutil.rmtree(源)
    源.mkdir()
    for n, b in 小清单:
        (源 / n).write_bytes(bytes([65 + 小清单.index((n, b))]) * b)


def 跑(参数, 目标目录=目标):
    环境 = dict(os.environ)
    环境["FT_MODEL_DIR"] = 转bash(目标目录)
    r = subprocess.run(["bash", 转bash(测试脚本)] + 参数,
                       capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=环境)
    return r.returncode, r.stdout + r.stderr


结果 = []


def 记(名, 通过, 附注=""):
    结果.append((名, 通过, 附注))
    print(f"  [{'通过' if 通过 else '失败'}] {名}{('　' + 附注) if 附注 else ''}")


def 大小(p):
    """文件不存在时返回 -1，避免断言直接把测试脚本炸掉。"""
    路径 = Path(p)
    return 路径.stat().st_size if 路径.is_file() else -1


# ------------------------------------------------------------ 场景 1 正常导入
print("\n########## 场景 1：正常导入 ##########")
造源()
rc, out = 跑([转bash(源)])
print(out[-700:])
记("源完整 -> 退出码 0", rc == 0, f"rc={rc}")
记("a/b/c 都已就位", all((目标 / n).is_file() for n, _ in 小清单))
记("字节数保持", all(大小(目标 / n) == b for n, b in 小清单))

# ------------------------------------------------------------ 场景 2 幂等
print("\n########## 场景 2：重复执行（幂等） ##########")
rc2, out2 = 跑([转bash(源)])
记("重复执行 -> 仍退出码 0", rc2 == 0, f"rc={rc2}")
记("重复执行没有报错", "阻断" not in out2)

# ------------------------------------------------------------ 场景 3 --verify
print("\n########## 场景 3：--verify ##########################")
rc3, out3 = 跑(["--verify"])
记("--verify -> 退出码 0", rc3 == 0, f"rc={rc3}")

# ------------------------------------------------------------ 场景 4 源缺文件
print("\n########## 场景 4：源缺文件 ##########")
造源()
(源 / "b.bin").unlink()
rc4, out4 = 跑([转bash(源)])
记("源缺文件 -> 退出码 1", rc4 == 1, f"rc={rc4}")
记("正确报出缺失的 b.bin", "b.bin" in out4)

# ------------------------------------------------------------ 场景 5 字节数不符
print("\n########## 场景 5：源文件字节数不符 ##########")
造源()
(源 / "b.bin").write_bytes(b"X" * 1999)   # 少 1 字节
rc5, out5 = 跑([转bash(源), "--fresh"])
记("字节数不符 -> 退出码 1", rc5 == 1, f"rc={rc5}")
记("正确报出字节数不符", "1999" in out5 and "2000" in out5)

# ------------------------------------------------------------ 场景 6 --fresh
print("\n########## 场景 6：--fresh 清空残片后导入 ##########")
造源()
(目标 / "a.bin").write_bytes(b"R" * 10)          # 制造残片
(目标 / "leftover.garbage").write_bytes(b"G" * 50)  # 制造垃圾文件
rc6, out6 = 跑([转bash(源), "--fresh"])
记("--fresh -> 退出码 0", rc6 == 0, f"rc={rc6}")
记("残片已清掉（a.bin 是新的 1000 字节）", 大小(目标 / "a.bin") == 1000)
记("垃圾文件已被清掉", not (目标 / "leftover.garbage").exists())

# ------------------------------------------------------------ 场景 7 --clean-src
print("\n########## 场景 7：--clean-src ##########################")
造源()
rc7, out7 = 跑([转bash(源), "--clean-src"])
记("--clean-src -> 退出码 0", rc7 == 0, f"rc={rc7}")
记("源目录已被删除", not 源.exists())

# ------------------------------------------------------------ 场景 8 无参数
print("\n########## 场景 8：无参数 ##########")
rc8, out8 = 跑([])
记("无参数 -> 退出码 2 且给出提示", rc8 == 2, f"rc={rc8}")

# ------------------------------------------------------------ 汇总
失败 = sum(1 for _, p, _ in 结果 if not p)
print("\n" + "=" * 60)
print(f" 结果：{len(结果) - 失败}/{len(结果)} 通过")
print("=" * 60)
shutil.rmtree(临时根, ignore_errors=True)
sys.exit(1 if 失败 else 0)
