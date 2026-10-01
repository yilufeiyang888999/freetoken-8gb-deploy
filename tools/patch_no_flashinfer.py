#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FreeToken × flashinfer 解耦补丁

背景
----
FreeToken 的 layers/activation.py / layers/norm.py / layers/rotary.py /
engine/sample.py 都用同一个开关做二选一：

    if is_flashinfer_installed():
        from flashinfer import ...                  # 需 nvcc 做 JIT
    else:
        from freetoken.kernel.triton.xxx import ... # 纯 triton，零依赖

本机 flashinfer 0.6.18.post1 能 import，于是开关判定为「可用」，
但 PyTorch 的 CUDA wheel 只带运行时库、不带 nvcc，
首次调用就在 CUDA graph 捕获阶段炸：

    RuntimeError: Could not find nvcc and default
    cuda_home='/usr/local/cuda' doesn't exist

做法
----
把 freetoken/kernel/backend.py 里的 is_flashinfer_installed() 改成恒返回 False。
一行改动，全局生效：activation / norm / rotary / sampling 全部落到内置 triton。

不受影响的部分（已核实）
------------------------
* attention  : 启动参数已显式 --attention-backend triton，与 flashinfer 无关
* MoE experts: 日志显示 "MoE experts: nvfp4 via triton"，同样无关
* b12x 路径  : nvfp4.py 由 _flashinfer_b12x_unusable() 门控，开关变 False 后
               直接返回 "flashinfer is not installed"，本来也不会被自动选中

附带动作
--------
打完补丁后额外预热一次 triton 版的 silu_and_mul，
把 kernel 编译进 ~/.triton/cache，避免它拖到 CUDA graph 捕获期间才首次编译。

回滚
----
    cp <...>/freetoken/kernel/backend.py.bak-flashinfer <...>/freetoken/kernel/backend.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

VENV = Path.home() / "freetoken-venv"
SITE = VENV / "lib" / "python3.12" / "site-packages"
PY = str(VENV / "bin" / "python")
BACKEND_PY = SITE / "freetoken" / "kernel" / "backend.py"
MARK = "FT_PATCH_NO_FLASHINFER"

LINE = "-" * 68


def banner(text: str) -> None:
    print(LINE)
    print(text)
    print(LINE)


def main() -> int:
    banner("FreeToken x flashinfer 解耦补丁")

    # ---------- 0. 前置检查 ----------
    if not Path(PY).is_file():
        print("[失败] 找不到 venv 里的 python：" + PY)
        return 1
    if not BACKEND_PY.is_file():
        print("[失败] 找不到 " + str(BACKEND_PY))
        print("       请把下面这条命令的输出贴回，助手会给出精确路径：")
        print("       find / -path '*freetoken/kernel/backend.py' 2>/dev/null")
        return 1
    print("[通过] venv python : " + PY)
    print("[通过] 目标文件    : " + str(BACKEND_PY))

    # ---------- 1. 打补丁 ----------
    banner("步骤 1 / 3　改写开关")
    src = BACKEND_PY.read_text(encoding="utf-8")

    if MARK in src:
        print("[跳过] 补丁早已打过，无需重复。")
    else:
        pattern = re.compile(
            r"(def\s+is_flashinfer_installed\s*\([^)]*\)[^:\n]*:[ \t]*\n)"  # 函数签名
            r"([ \t]+)"                                                     # 缩进
            r"(return\s+[^\n]+)"                                            # 原 return
        )
        hit = pattern.search(src)
        if hit is None:
            print("[失败] 没能自动定位 is_flashinfer_installed 的函数体。")
            print("       请把下面这条命令的输出贴回，助手会给出精确改法：")
            print('       grep -n -A 6 "def is_flashinfer_installed" ' + str(BACKEND_PY))
            return 2

        original = hit.group(3).strip()
        backup = BACKEND_PY.with_name(BACKEND_PY.name + ".bak-flashinfer")
        if not backup.exists():
            shutil.copy2(BACKEND_PY, backup)
            print("[备份] " + str(backup))

        patched = (
            src[: hit.start(3)]
            + hit.group(2) + "return False  # " + MARK + "  (原: " + original + ")"
            + src[hit.end(3):]
        )
        BACKEND_PY.write_text(patched, encoding="utf-8")
        print("[改写] " + original)
        print("  -->    return False   # " + MARK)

    # ---------- 2. 验证开关 ----------
    banner("步骤 2 / 3　验证开关已生效")
    check = (
        "from freetoken.kernel.backend import is_flashinfer_installed as f\n"
        "print('is_flashinfer_installed() =', f())\n"
        "assert f() is False, '补丁未生效，仍然认为 flashinfer 可用'\n"
        "print('OK: freetoken 不会再走 flashinfer')\n"
    )
    proc = subprocess.run([PY, "-c", check], capture_output=True, text=True)
    print((proc.stdout + proc.stderr).strip())
    if proc.returncode != 0:
        print("[失败] 验证未通过，先不要启动服务。")
        return 3

    # ---------- 3. 预热 triton 激活核 ----------
    banner("步骤 3 / 3　预热 triton silu_and_mul（首次编译，5~30 秒）")
    warm = (
        "import torch\n"
        "from freetoken.layers.activation import silu_and_mul\n"
        "x = torch.randn(16, 4096, device='cuda', dtype=torch.bfloat16)\n"
        "out = torch.empty(16, 2048, device='cuda', dtype=torch.bfloat16)\n"
        "ret = silu_and_mul(x, out=out)\n"
        "print('OK: triton silu_and_mul 实跑通过 ->', tuple(out.shape), out.dtype)\n"
        "print('OK: 返回值 is out ->', ret is out)\n"
        "print('OK: 非空 ->', bool(out.abs().sum().item() > 0))\n"
    )
    proc = subprocess.run([PY, "-c", warm], capture_output=True, text=True)
    print((proc.stdout + proc.stderr).strip())
    if proc.returncode != 0:
        print()
        print("[注意] 预热失败，但补丁本体已经生效。")
        print("       启动时如果又报错，把上面这段贴回即可。")

    # ---------- 收尾 ----------
    banner("完成")
    print("下一步，直接启动：")
    print("    bash ~/serve_freetoken.sh")
    print()
    print("要回滚（恢复访问 flashinfer）：")
    print("    cp " + str(BACKEND_PY) + ".bak-flashinfer " + str(BACKEND_PY))
    return 0


if __name__ == "__main__":
    sys.exit(main())
