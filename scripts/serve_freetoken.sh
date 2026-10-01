#!/usr/bin/env bash
# ==============================================================================
# FreeToken 启动脚本 —— Qwen3.6-35B-A3B-NVFP4（WSL2 + RTX 5060 Laptop 8G）
# ------------------------------------------------------------------------------
# 为什么需要这个脚本，而不是直接敲 ft serve：
#
#   WSL 下有三道彼此独立的天花板，任何一个没算到都会让启动失败，而且失败点
#   全都在「读完 22GB 权重之后」，报错还长得像别的问题。本脚本在真正启动前
#   把它们逐个检查掉，并且让该失败的事**几秒钟内就失败**。
#
#   ① GPU 显存      —— hybrid 下非专家权重 + KV cache + slot cache 都在这里
#   ② 锁页内存       —— expert banks 要被 GPU 搬运的部分必须 cudaHostRegister，
#                      WSL 的配额是「物理内存 × 40%」。本机 27.4 × 40% = 11.0 GiB，
#                      而本模型 banks 要 16.93 GiB → 超出。
#                      解法：--moe-cpu-layers auto，把装不下的层挪到 CPU 上算。
#                      （源码：engine.py::_pin_budget_bytes / _auto_cpu_layers）
#   ③ mlock 配额     —— 挪到 CPU 的那些层改用 mlock 锁住，但 mlock 受
#                      RLIMIT_MEMLOCK（`ulimit -l`）限制。受限时降级为 pageable，
#                      能跑但可能被换出。本脚本会尝试提升。
#
#   另外还有一个与资源无关、纯粹是环境缺失的坑：
#     Triton 首次使用要把驱动辅助模块编译成 .so，需要 PATH 上有 gcc 或 clang。
#     Triton 源码里只认 CC 环境变量 → clang → gcc，**不读 sysconfig**。
#     它在 `_reset_moe_offload_cache()` 里才触发，也就是权重读完之后。
#     本脚本用一次 3 秒的预检把它提前暴露。
#
#   ④ attention 后端 —— 这条决定了要不要装 CUDA toolkit，单独说：
#      引擎的 auto 按 trtllm → fa,fi → fi → triton 的顺序挑（engine.py:127）。
#      sm_120（RTX 50 系）上前两个的架构条件都不成立，于是落到 **fi（flashinfer）**。
#      而 fi 会在捕获 CUDA graph 时用 **nvcc 现场编译 attention 模块**，
#      要求 CUDA_HOME 下有 bin/nvcc、include/、lib64 —— 也就是完整 CUDA toolkit。
#      缺了就在「读完 22GB 权重 + 建完 banks」之后报
#      `Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist`。
#
#      triton 后端（BackendInfo 里没有 requires_flashinfer）是纯 Triton 实现，
#      不需要 nvcc。官方在 engine.py:210 的报错正文里就推荐它。
#      本机的瓶颈是 15/40 层专家在 CPU 上算，注意力不是瓶颈，
#      所以默认走 triton：**零下载，少一个 3~4 GB 的 CUDA toolkit 依赖**。
#      想换回 flashinfer：--attention-backend fi（需先装 cuda-toolkit-13-0）。
#
# 注意：脚本注释用中文，但**变量名一律 ASCII** —— bash 的标识符只允许
#       [a-zA-Z_][a-zA-Z0-9_]*，中文变量名会被当成命令名报 command not found。
#
#   ⑤ CORS 白名单 —— 浏览器网页端 / IDE 扩展要跨源访问时，服务默认只放行
#      Tauri 桌面端的三个 origin（args.py:68）：
#        tauri://localhost,http://tauri.localhost,http://localhost:1420
#      所以从 file:// 打开的 HTML（Origin: null）或别的本地端口会被浏览器拦掉。
#      本脚本默认传 --cors-origins '*'（源码 api_server.py:412 会把 '*' 展开成
#      allow_origins=["*"]）。服务只监听 127.0.0.1，不对局域网开放，风险可控。
#      想收紧：--cors-origins 'http://127.0.0.1:8080,http://localhost:8080'
#
# 用法：
#   bash serve_freetoken.sh               # 前台启动，日志同时写文件
#   bash serve_freetoken.sh --dry-run     # 只做检查并打印命令，不启动
#   bash serve_freetoken.sh --skip-preflight         # 跳过运行时预检（缓存已热时可省几秒）
#   bash serve_freetoken.sh --attention-backend fi   # 换回 flashinfer（需 CUDA toolkit）
#   bash serve_freetoken.sh --cors-origins ''        # 关掉 CORS 头（只用 curl / IDE 时）
#   环境变量 FT_ATTN_BACKEND / FT_CORS_ORIGINS 可替代对应开关
# ==============================================================================

set -uo pipefail

# 本脚本所在目录（仓库 scripts/），用于定位同仓库的配套工具
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VENV_DIR="${HOME}/freetoken-venv"
FT_BIN="${VENV_DIR}/bin/ft"
VENV_PY="${VENV_DIR}/bin/python"
MODEL_DIR="${HOME}/models/Qwen3.6-35B-A3B-NVFP4"
LOG_FILE="${HOME}/models/freetoken-serve.log"
HOST="127.0.0.1"
PORT="1919"
MAX_SEQ_LEN="16384"

# ---- 启动后自动调整 KV 容量（0 = 不动）--------------------------------------
# 关键：--max-seq-len-override 只改「声明值」，**不改实际分配**。KV cache 的
# 真实容量由显存预算决定 —— 出厂切分只给 8268 token，prompt 超过直接 HTTP 400，
# 哪怕 /v1/models 信誓旦旦地说 16384。要真正拿到长上下文，只能在服务起来之后
# 通过 /v1/cache/rebuild 重新切分（KV / MoE / Mamba 三块共享同一个池子）。
# 详见《FreeToken-部署实录.md》11.2；调优工具 ft_cache_tune.py。
KV_CONTEXT="${FT_KV_CONTEXT:-16384}"
KV_MOE="${FT_KV_MOE:-1108}"
KV_MAMBA="${FT_KV_MAMBA:-20}"

DRY_RUN=0
SKIP_PREFLIGHT=0
ATTN_BACKEND="${FT_ATTN_BACKEND:-triton}"
CORS_ORIGINS="${FT_CORS_ORIGINS:-*}"
SITE_PKGS=""
HAS_ATTN_ARG=0
HAS_CORS_ARG=0

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)                DRY_RUN=1 ;;
        --skip-preflight)         SKIP_PREFLIGHT=1 ;;
        --attention-backend)      shift
                                  if [ $# -lt 1 ]; then
                                      printf "  --attention-backend 后面缺参数\n"; exit 2
                                  fi
                                  ATTN_BACKEND="$1" ;;
        --attention-backend=*)    ATTN_BACKEND="${1#*=}" ;;
        --cors-origins)           shift
                                  if [ $# -lt 1 ]; then
                                      printf "  --cors-origins 后面缺参数\n"; exit 2
                                  fi
                                  CORS_ORIGINS="$1" ;;
        --cors-origins=*)         CORS_ORIGINS="${1#*=}" ;;
        --kv-context)             shift
                                  if [ $# -lt 1 ]; then
                                      printf "  --kv-context 后面缺参数\n"; exit 2
                                  fi
                                  KV_CONTEXT="$1" ;;
        --kv-context=*)           KV_CONTEXT="${1#*=}" ;;
        --kv-moe)                 shift
                                  if [ $# -lt 1 ]; then
                                      printf "  --kv-moe 后面缺参数\n"; exit 2
                                  fi
                                  KV_MOE="$1" ;;
        --kv-mamba)               shift
                                  if [ $# -lt 1 ]; then
                                      printf "  --kv-mamba 后面缺参数\n"; exit 2
                                  fi
                                  KV_MAMBA="$1" ;;
        *)                        printf "  未知参数：%s\n" "$1"; exit 2 ;;
    esac
    shift
done

# 这里只放本脚本真正验证过、且对这套模型有意义的三个值。
# auto 交给引擎自己挑（sm_120 上会选 fi，因此同样需要 CUDA toolkit）。
case "${ATTN_BACKEND}" in
    triton|fi|auto) ;;
    *) printf "  不支持的 --attention-backend：%s（可选：triton / fi / auto）\n" "${ATTN_BACKEND}"
       exit 2 ;;
esac

# KV 几何三个值都得是数字，否则后面的 curl 会拼出畸形 JSON
for PAIR in "KV_CONTEXT:${KV_CONTEXT}" "KV_MOE:${KV_MOE}" "KV_MAMBA:${KV_MAMBA}"; do
    case "${PAIR#*:}" in
        ''|*[!0-9]*) printf "  %s 需要非负整数，实得：%s\n" "${PAIR%%:*}" "${PAIR#*:}"
                     exit 2 ;;
    esac
done

step() { printf "\n%s\n %s\n%s\n" "────────────────────────────────────────────────────────────────" "$1" "────────────────────────────────────────────────────────────────"; }
ok()   { printf "  [通过] %s\n" "$1"; }
warn() { printf "  [警告] %s\n" "$1"; }
bad()  { printf "  [阻断] %s\n" "$1"; }
info() { printf "  %s\n" "$1"; }

printf "\n%s\n FreeToken 启动\n%s\n" \
  "════════════════════════════════════════════════════════════════" \
  "════════════════════════════════════════════════════════════════"

# ================================================================ 步骤 1 环境
step "步骤 1 / 4　环境检查"

if [ ! -x "${FT_BIN}" ]; then
    if command -v ft >/dev/null 2>&1; then
        FT_BIN="$(command -v ft)"
        warn "未找到 ${VENV_DIR}/bin/ft，改用 PATH 上的 ${FT_BIN}"
    else
        bad "找不到 ft 可执行文件"
        info "请先执行：source ${VENV_DIR}/bin/activate"
        exit 1
    fi
fi
ok "ft：$("${FT_BIN}" --version 2>&1 | head -1)"

if [ ! -x "${VENV_PY}" ]; then
    bad "找不到虚拟环境 Python：${VENV_PY}"
    exit 1
fi

if [ ! -d "${MODEL_DIR}" ]; then
    bad "模型目录不存在：${MODEL_DIR}"
    exit 1
fi
FILE_COUNT="$(find "${MODEL_DIR}" -maxdepth 1 -type f | wc -l)"
if [ "${FILE_COUNT}" -lt 17 ]; then
    warn "模型目录只有 ${FILE_COUNT} 个文件，期望 17 个 —— 可能没复制全"
else
    ok "模型目录 ${FILE_COUNT} 个文件"
fi

# --- C 编译器：Triton 运行时编译的硬依赖，缺了就必炸，而且是在最后一步炸 ---
# Triton 的查找顺序（runtime/build.py::_build）：os.environ["CC"] -> clang -> gcc
if [ -n "${CC:-}" ] && ! command -v "${CC}" >/dev/null 2>&1; then
    bad "环境变量 CC=${CC} 指向的命令不存在 —— Triton 会直接用这个值，不看 gcc"
    info "处理：unset CC 让 Triton 自己找 gcc，或把 CC 改成正确路径"
    exit 1
fi
CC_FOUND=""
if [ -n "${CC:-}" ]; then
    CC_FOUND="${CC}"
elif command -v gcc >/dev/null 2>&1; then
    CC_FOUND="$(command -v gcc)"
elif command -v clang >/dev/null 2>&1; then
    CC_FOUND="$(command -v clang)"
    export CC="clang"
fi
if [ -z "${CC_FOUND}" ]; then
    bad "PATH 上找不到 C 编译器（gcc / clang）"
    info "Triton 要把驱动辅助模块编译成 .so，这一步没编译器必失败。装："
    info "    sudo apt update && sudo apt install -y build-essential"
    info "（只装 gcc 也行，但 build-essential 会一并带上 libc6-dev，标准头文件少不了）"
    exit 1
fi
ok "C 编译器：${CC_FOUND}$([ -n "${CC:-}" ] && echo '（来自 CC 环境变量）')"

# --- flashinfer 解耦自检 ---------------------------------------------------
# freetoken 的 activation / norm / rotary / sampling 都通过 is_flashinfer_installed()
# 做二选一：真 → 走 flashinfer（JIT 要 nvcc，本机没有）；假 → 走自带的 triton 实现。
# 补丁会随 venv 重装 / freetoken 升级丢掉，所以每次启动都查一遍，
# 免得将来又在「读完 22GB 之后」才炸，白等一分钟。
SITE_PKGS_FI=""
for CAND in "${VENV_DIR}"/lib/python3.*/site-packages; do
    if [ -d "${CAND}/freetoken" ]; then SITE_PKGS_FI="${CAND}"; break; fi
done
FI_SWITCH="${SITE_PKGS_FI}/freetoken/kernel/backend.py"
PATCH_SCRIPT="${SCRIPT_DIR}/../tools/patch_no_flashinfer.py"
if [ -z "${SITE_PKGS_FI}" ] || [ ! -f "${FI_SWITCH}" ]; then
    info "未定位到 freetoken/kernel/backend.py，跳过 flashinfer 解耦自检"
elif grep -q "FT_PATCH_NO_FLASHINFER" "${FI_SWITCH}" 2>/dev/null; then
    ok "flashinfer 已解耦（is_flashinfer_installed() 恒为 False，全走内置 triton）"
else
    warn "flashinfer 仍处于启用状态 —— 会在捕获 CUDA graph 时因缺 nvcc 失败"
    info "症状：RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda'"
    info "     栈里出现 freetoken/layers/activation.py（不是 attention/fi.py）"
    info "换 --attention-backend 解决不了这个，解耦开关是唯一的零下载解法："
    if [ -f "${PATCH_SCRIPT}" ]; then
        info "    ${VENV_PY} ${PATCH_SCRIPT}"
    else
        info "    重跑 flashinfer 解耦补丁（patch_no_flashinfer.py）"
    fi
fi

# ================================================================ 步骤 2 锁页预算
step "步骤 2 / 4　内存预算体检"

if grep -qi microsoft /proc/version 2>/dev/null; then
    TOTAL_KB="$(awk '/MemTotal/{print $2}' /proc/meminfo)"
    TOTAL_GB="$(awk -v k="${TOTAL_KB:-0}" 'BEGIN{printf "%.1f", k/1024/1024}')"
    BUDGET_GB="$(awk -v k="${TOTAL_KB:-0}" 'BEGIN{printf "%.1f", k/1024/1024*0.4}')"
    info "WSL 物理内存 ${TOTAL_GB} GiB"
    info "引擎的锁页预算 = 40% ≈ ${BUDGET_GB} GiB（本模型 banks 约 16.9 GiB，超出）"
    ok "已启用 --moe-cpu-layers auto，装不下的层会挪到 CPU"

    # --- mlock 配额：被挪到 CPU 的 banks 要用 mlock 锁住 ---
    # 注意：不支持 -l 的平台（如 MSYS/Git Bash）返回**空字符串**而不是报错，
    # 不先判断空值的话会一路走到「提升失败」分支，给出误导性的警告。
    ML_SOFT="$(ulimit -Sl 2>/dev/null || true)"
    ML_HARD="$(ulimit -Hl 2>/dev/null || true)"
    if [ -z "${ML_SOFT}" ]; then
        info "本 shell 不支持 ulimit -l 查询，跳过 mlock 配额检查"
    else
        info "锁页配额 ulimit -l：软限 ${ML_SOFT} KB，硬限 ${ML_HARD} KB"
        if [ "${ML_SOFT}" = "unlimited" ]; then
            ok "mlock 配额充足，CPU 层 banks 会被真正锁住"
        elif ulimit -l unlimited 2>/dev/null; then
            ok "mlock 配额已提升为 unlimited"
        elif sudo -n prlimit --memlock=unlimited:unlimited --pid "$$" 2>/dev/null; then
            # 只对「当前这个脚本进程」提限，它随后 fork 的 ft 会继承。
            # 能走到这里说明 sudo 免密可用（或凭证已缓存）。
            ok "已通过 sudo 把 mlock 配额提升为 unlimited"
        else
            warn "mlock 配额不足 —— CPU 层 banks 会降级为 pageable"
            info "不阻断启动：它们仍由 CPU 计算，只是内存吃紧时可能被换出，decode 会卡顿。"
            info "硬限也只有 ${ML_HARD} KB，非 root 提不上去，必须借管理员权限。二选一："
            info "  a)【一次性】在准备敲命令的这个终端里，先执行下面这条，再重跑本脚本："
            info '       sudo prlimit --memlock=unlimited:unlimited --pid $$'
            info '     （$$ 是当前 shell 的 PID；脚本是它的子进程，会继承新配额）'
            ML_USER="$(id -un 2>/dev/null || true)"
            [ -n "${ML_USER}" ] || ML_USER="${USER:-<用户名>}"
            info "  b)【永久】/etc/security/limits.conf 末尾加一行，再在 Windows 侧"
            info "       wsl --shutdown 重进（PAM 才会重新套用该文件）："
            info "       ${ML_USER} - memlock unlimited"
        fi
    fi
else
    warn "未检测到 WSL 内核标记 —— 若不是 WSL，请去掉 --moe-cpu-layers（Linux 不设锁页配额）"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    FREE_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)"
    if [ -n "${FREE_MIB}" ]; then
        FREE_GB="$(awk -v m="${FREE_MIB}" 'BEGIN{printf "%.2f", m/1024}')"
        info "当前显存空闲 ${FREE_GB} GiB"
        if [ "${FREE_MIB}" -lt 6500 ]; then
            warn "空闲显存偏少，建议先关掉浏览器 / WPS / 微信再启动"
        fi
    fi
fi

# ================================================================ 步骤 3 运行时预检
step "步骤 3 / 4　运行时预检"

if [ "${SKIP_PREFLIGHT}" -eq 1 ]; then
    info "已按参数跳过"
else
    # 直接调用出错的那一行：_reset_cache_gpu -> triton.jit kernel -> driver.active
    # driver.active 会触发 CudaUtils()，也就是需要 C 编译器的那次编译。
    # 提前跑一次，缺编译器时 3 秒内就报出来，而不是等读完 22GB 权重。
    PREFLIGHT_OUT="$("${VENV_PY}" -c 'import torch
from triton.runtime import driver
driver.active.get_current_device()
print("[通过] Triton 驱动模块编译 / 加载成功")' 2>&1)"
    PREFLIGHT_RC=$?
    printf '%s\n' "${PREFLIGHT_OUT}" | sed 's/^/  /'
    if [ "${PREFLIGHT_RC}" -ne 0 ]; then
        bad "Triton 运行时不可用，ft serve 一定会在同一个地方失败"
        info "常见原因："
        info "  1) 缺 C 编译器（见上方报错）→ sudo apt install -y build-essential"
        info "  2) libcuda.so.1 找不到 → 确认 /usr/lib/wsl/lib/libcuda.so.1 存在"
        info "     不存在说明 Windows 侧显卡驱动太老，升到 555.85 以上"
        info "  3) Python 头文件缺失（缺 Python.h）→ sudo apt install -y python3-dev"
        exit 1
    fi
    ok "Triton 可正常编译并加载（结果已缓存，ft serve 不会重复编译）"
fi

# --- attention 后端的前置检查（不放上面那个 if 里：--skip-preflight 只该跳过
#     耗时最长的 Triton 编译，不该把「注定要炸」的后端依赖一起跳过）---

# 先确认「本机装的这套 freetoken」认不认这些参数和取值。
# nightly 版本之间参数集会变，静态查文件比跑 --help 快，也不用 import torch。
for CAND in "${VENV_DIR}"/lib/python3.*/site-packages; do
    if [ -d "${CAND}/freetoken" ]; then SITE_PKGS="${CAND}"; break; fi
done
if [ -n "${SITE_PKGS}" ] && grep -q -- '"--attention-backend"' "${SITE_PKGS}/freetoken/server/args.py" 2>/dev/null; then
    HAS_ATTN_ARG=1
fi

if [ -z "${SITE_PKGS}" ]; then
    warn "没在 ${VENV_DIR} 下定位到 site-packages，跳过 attention 后端参数自检"
elif [ "${HAS_ATTN_ARG}" -eq 0 ]; then
    warn "本机这套 freetoken 没有 --attention-backend 参数，无法切换 attention 后端"
    info "它只认引擎自行挑选的 fi（flashinfer），那就必须先装 CUDA toolkit："
    info "  sudo apt install -y cuda-toolkit-13-0   （先配好 NVIDIA 的 WSL CUDA 源）"
    ATTN_BACKEND="__unsupported__"
elif [ "${ATTN_BACKEND}" != "auto" ] && \
     ! grep -q "\"${ATTN_BACKEND}\"" "${SITE_PKGS}/freetoken/attention/__init__.py" 2>/dev/null; then
    warn "本机这套 freetoken 的后端注册表里没有 '${ATTN_BACKEND}'，无法选择"
    ATTN_BACKEND="__unsupported__"
fi

if [ "${ATTN_BACKEND}" = "__unsupported__" ]; then
    # 不阻断：让它照原样启动，完整日志还能提供更多线索
    info "本次将不带 --attention-backend 启动（等同让引擎自行选择）"
elif [ "${ATTN_BACKEND}" = "triton" ]; then
    ok "attention 后端 triton —— 纯 Triton 实现，不需要 nvcc / CUDA toolkit"
else
    # fi / auto 都会用到 flashinfer，它的 JIT 需要 CUDA_HOME 下有 nvcc。
    # 预检直接调用出错的那一行（flashinfer/jit/cpp_ext.py::get_cuda_path），
    # 让它在 3 秒内暴露，而不是等读完 22GB 权重、建完 banks 之后。
    if FI_OUT="$("${VENV_PY}" -c 'from flashinfer.jit.cpp_ext import get_cuda_path; print("CUDA_HOME =", get_cuda_path())' 2>&1)"; then
        ok "flashinfer 的 CUDA 工具链就绪（${FI_OUT##*= }）"
    else
        printf '%s\n' "${FI_OUT}" | sed 's/^/  /'
        bad "attention 后端 ${ATTN_BACKEND} 依赖 flashinfer，而它找不到 CUDA 工具链"
        info "flashinfer 编译 attention 模块时要求 CUDA_HOME 下有"
        info "bin/nvcc、include/、lib64 —— 是完整 CUDA toolkit，不是驱动自带的运行时。"
        info "两条路："
        info "  a)【推荐 · 零下载】改用 Triton 实现，重跑："
        info "       bash $0 --attention-backend triton"
        info "  b)【保留 flashinfer】装 CUDA toolkit（约 3~4 GB）："
        info "       wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
        info "       sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt update"
        info "       sudo apt install -y cuda-toolkit-13-0"
        info "     版本要和 torch 的 CUDA 大版本对齐（当前为 cu13）；"
        info "     源里没有这个包名就用 apt-cache search '^cuda-toolkit-13' 查实际名。"
        exit 1
    fi
fi

# --- CORS 白名单自检（浏览器网页端 / IDE 扩展需要）---
if [ -z "${SITE_PKGS}" ]; then
    warn "未定位 site-packages，跳过 --cors-origins 自检"
elif grep -q -- '"--cors-origins"' "${SITE_PKGS}/freetoken/server/args.py" 2>/dev/null; then
    HAS_CORS_ARG=1
    if [ "${CORS_ORIGINS}" = "*" ]; then
        ok "CORS 已放开（--cors-origins '*'）—— 浏览器可直连，服务仅监听 ${HOST}"
    elif [ -z "${CORS_ORIGINS}" ]; then
        info "CORS 已关闭（--cors-origins ''）—— 仅 curl / IDE 服务端调用可用"
    else
        ok "CORS 白名单：${CORS_ORIGINS}"
    fi
else
    warn "本机这套 freetoken 没有 --cors-origins 参数"
    info "浏览器网页端会被 CORS 拦（默认只放行 Tauri 桌面端的 3 个 origin）。"
    info "对策：改用系统代理 / 中间网关转发，或升级 freetoken。"
fi

# ================================================================ 步骤 4 启动
step "步骤 4 / 4　启动服务"

ARGS=(
    serve
    --model "${MODEL_DIR}"
    --text-model-only
    --moe-strategy auto
    --moe-cpu-layers auto
    --max-seq-len-override "${MAX_SEQ_LEN}"
    --host "${HOST}"
    --port "${PORT}"
    --enable-cache-report
)

# 只有前面自检确认「本机这套 freetoken 认得该参数」时才传，避免喂给它一个
# 不存在的参数直接以 "unrecognized arguments" 退出
if [ "${HAS_CORS_ARG}" -eq 1 ]; then
    # 注意：空串也要照样传 —— freetoken 的 '' 表示「完全关掉 CORS 中间件」，
    # 与「不传该参数」（保留 Tauri 默认白名单）是两种不同行为。
    ARGS+=(--cors-origins "${CORS_ORIGINS}")
fi

if [ "${ATTN_BACKEND}" != "__unsupported__" ]; then
    ARGS+=(--attention-backend "${ATTN_BACKEND}")
fi

info "命令："
printf '    %s' "${FT_BIN}"
for ONE in "${ARGS[@]}"; do printf ' %s' "${ONE}"; done
printf '\n'
info "日志：${LOG_FILE}"
info ""
if [ "${KV_CONTEXT}" -gt 0 ]; then
    info "【自动调优】服务就绪后会后台把 KV 容量调到 ${KV_CONTEXT} token。"
    info "    为什么需要：出厂切分只给 KV 8268 token，而 /v1/models 声明 16384 ——"
    info "    不调的话，prompt 超过 8268 直接 HTTP 400（见文档《FreeToken-部署实录.md》11.2）。"
    info "    不想动这项：--kv-context 0。执行结果写在日志末尾。"
else
    info "【自动调优】已关闭（--kv-context 0），KV 保持出厂切分（约 8268 token）。"
fi

if [ "${DRY_RUN}" -eq 1 ]; then
    info "--dry-run：只检查，不实际启动"
    exit 0
fi

info ""
info "启动约 1~2 分钟（读 22GB 权重 + 串行建 banks），中途无进度属正常。"
info "注意 'Uvicorn running on http://${HOST}:${PORT}' 出现得很早，**它不是就绪标志** ——"
info "   那时模型还没加载，后端 worker 可能随后就退出。真正可以发请求的界标是："
info "   日志走过 'Free memory after initialization' 且没有出现 Backend worker is gone。"
info "要盯的几行："
info "  'Auto-selected attention backend: ...'                             ← 期望 triton"
info "  '--moe-cpu-layers auto: banks ... locking N head+tail MoE layers'  ← N 是挪到 CPU 的层数"
info "  'Free memory after initialization: X GiB'                          ← 低于 0.3 后面易 OOM"
MODEL_NAME="$(basename "${MODEL_DIR}")"
info ""
info "服务起来后（另开一个终端）："
info "  1) 确认就绪 —— 能列出模型才算真的好了："
info "     curl -s http://${HOST}:${PORT}/v1/models"
info ""
info "  2) 提问。**Qwen3.6 默认开思考模式**，思维链会先吃掉大量 token；"
info "     max_tokens 给小了会看到空答案（content 为空、finish_reason=length），"
info "     那不是故障。ChatBI / 结构化输出这类任务建议关掉思考："
info "     curl -s http://${HOST}:${PORT}/v1/chat/completions \\"
info "       -H 'Content-Type: application/json' \\"
info "       -d '{\"model\":\"${MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}],\"chat_template_kwargs\":{\"enable_thinking\":false},\"max_tokens\":256}'"
info "     要保留思考就把 chat_template_kwargs 那段去掉，并把 max_tokens 提到 2048 以上。"
info ""
info "  3) 浏览器聊天界面（见文档《FreeToken-部署实录.md》第 10 章）："
if [ "${HAS_CORS_ARG}" -eq 1 ] && [ "${CORS_ORIGINS}" = "*" ]; then
    info "     用浏览器打开仓库里的 chat.html 即可（CORS 已放开）："
    CHAT_WIN="$(wslpath -w "${SCRIPT_DIR}/../tools/chat.html" 2>/dev/null || true)"
    info "       ${CHAT_WIN:-<仓库>/tools/chat.html}"
else
    info "     本次 CORS 未放开，浏览器直连会被拦；需要时用 --cors-origins '*' 重启。"
fi
info ""
info "杀掉服务：Ctrl+C，或另一终端 ft down"
info ""

mkdir -p "$(dirname "${LOG_FILE}")"

# 小显存卡上碎片化很致命（PyTorch 自己也会建议开这个）
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---- KV 几何调优：服务就绪后自动执行 ----------------------------------------
# 必须在服务起来**之后**才能调（/v1/cache/rebuild 是运行时接口），所以丢到后台
# 轮询等待，绝不阻塞前台服务。等服务就绪的判据用 /v1/cache/status —— 它比
# /v1/models 晚得多，Uvicorn 起监听时模型往往还没加载完。
if [ "${KV_CONTEXT}" -gt 0 ] && command -v curl >/dev/null 2>&1; then
    (
        waited=0
        while [ "${waited}" -lt 600 ]; do
            if curl -sf --max-time 3 "http://${HOST}:${PORT}/v1/cache/status" 2>/dev/null \
                 | grep -q '"state"'; then
                printf '\n[KV] 服务就绪，调整缓存几何 -> %s token\n' "${KV_CONTEXT}" \
                    >> "${LOG_FILE}"
                curl -s --max-time 300 -X POST "http://${HOST}:${PORT}/v1/cache/rebuild" \
                    -H 'Content-Type: application/json' \
                    -d "{\"num_pages\":${KV_CONTEXT},\"moe_cache_size\":${KV_MOE},\"num_mamba_slots\":${KV_MAMBA},\"mode\":\"if_idle\",\"timeout\":280}" \
                    >> "${LOG_FILE}" 2>&1
                printf '\n' >> "${LOG_FILE}"
                break
            fi
            sleep 5
            waited=$((waited + 5))
        done
    ) &
fi

# 前台跑 + 同步落日志：出错时把 ${LOG_FILE} 贴回来即可复盘。
# 注意不要用 exec —— exec 加上管道并不会真的替换当前进程，反而会让
# Ctrl+C 只杀掉 tee、留下孤儿 ft 进程占着显存。
"${FT_BIN}" "${ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
exit "${PIPESTATUS[0]}"
