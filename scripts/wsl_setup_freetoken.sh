#!/usr/bin/env bash
# ==============================================================================
# FreeToken WSL2 环境准备脚本
# ------------------------------------------------------------------------------
# 用途：在 WSL2 的 Ubuntu 里执行，把 FreeToken 跑起来所需的环节一次性备齐
# 用法：bash wsl_setup_freetoken.sh
#
# 设计说明：
#   1. 可重复执行，已完成的步骤会自动跳过。
#   2. 只做安装和检测，不修改系统级配置（CUDA toolkit 等需你确认的会打印指引）。
#   3. 检测结论按「阻断 / 警告 / 通过」三级标注，与 freetoken_deploy.py 保持一致。
# ==============================================================================

set -uo pipefail

PASS=0
WARN=0
BLOCK=0
BLOCKERS=""

# ------------------------------------------------------------------ 输出工具
step()  { printf "\n%s\n %s\n%s\n" "────────────────────────────────────────────────────────────────" "$1" "────────────────────────────────────────────────────────────────"; }
ok()    { printf "  [通过] %s\n" "$1"; PASS=$((PASS + 1)); }
warn()  { printf "  [警告] %s\n" "$1"; WARN=$((WARN + 1)); }
block() { printf "  [阻断] %s\n" "$1"; BLOCK=$((BLOCK + 1)); BLOCKERS="${BLOCKERS}    - $1\n"; }
info()  { printf "  %s\n" "$1"; }

# 需要的 Python 版本：正式包 >= 3.10；nightly 预编译内核包锁定 3.12
VENV_DIR="${HOME}/freetoken-venv"
PY_VERSION="3.12"

printf "\n%s\n FreeToken WSL2 环境准备\n%s\n" "════════════════════════════════════════════════════════════════" "════════════════════════════════════════════════════════════════"

# ================================================================ 步骤 1 运行环境
step "步骤 1 / 5　运行环境"

if grep -qi microsoft /proc/version 2>/dev/null; then
    ok "确认运行在 WSL 内"
else
    block "当前不在 WSL 环境。请先以管理员身份在 PowerShell 运行：wsl --install -d Ubuntu"
fi

# 必须排除 Docker Desktop 的专用发行版 —— 它没有正常用户态，装不了东西
DISTRO_NAME="${WSL_DISTRO_NAME:-未知}"
info "发行版名称：${DISTRO_NAME}"
case "${DISTRO_NAME}" in
    docker-desktop*)
        block "当前是 Docker Desktop 的专用发行版，不能当开发环境用。"
        info  "请另装一个通用发行版：wsl --install -d Ubuntu --location D:\\WSL\\Ubuntu"
        info  "装完用 wsl -d Ubuntu 进入，再重新运行本脚本"
        ;;
    *)
        ok "发行版可正常使用"
        ;;
esac

ARCH="$(uname -m)"
info "CPU 架构：${ARCH}"
if [ "${ARCH}" = "x86_64" ]; then
    ok "满足 FreeToken 的 x86_64 要求"
else
    block "FreeToken 仅支持 x86_64，当前为 ${ARCH}"
fi

if command -v python3 >/dev/null 2>&1; then
    info "python3：$(python3 --version 2>&1)"
else
    warn "未找到 python3。执行 sudo apt update && sudo apt install -y python3 python3-venv 安装"
fi

# ================================================================ 步骤 2 GPU 直通
step "步骤 2 / 5　GPU 直通（WSL2 最常见的踩坑点）"

# WSL2 的 CUDA 库不在默认 PATH 里，先补上
WSL_LIB="/usr/lib/wsl/lib"
if [ -d "${WSL_LIB}" ]; then
    case ":${PATH}:" in
        *":${WSL_LIB}:"*) ;;
        *) export PATH="${WSL_LIB}:${PATH}" ;;
    esac
    case ":${LD_LIBRARY_PATH:-}:" in
        *":${WSL_LIB}:"*) ;;
        *) export LD_LIBRARY_PATH="${WSL_LIB}:${LD_LIBRARY_PATH:-}" ;;
    esac
    ok "找到 WSL CUDA 库目录，已临时加入 PATH"
else
    warn "未找到 ${WSL_LIB}。若 nvidia-smi 不可用，请在 Windows 侧升级 NVIDIA 驱动后执行 wsl --shutdown"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_LINE="$(nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader 2>/dev/null | head -n1)"
    if [ -n "${GPU_LINE}" ]; then
        info "GPU：${GPU_LINE}"
        ok "GPU 直通正常"
    else
        block "nvidia-smi 无输出，GPU 直通异常"
    fi
else
    block "WSL 内未找到 nvidia-smi。请在 Windows 侧确认驱动已安装，然后执行 wsl --shutdown 重启"
fi

# ================================================================ 步骤 3 资源配额
step "步骤 3 / 5　资源配额（内存 + 磁盘）"

MEM_TOTAL_KB="$(awk '/MemTotal/{print $2}' /proc/meminfo 2>/dev/null || echo 0)"
MEM_TOTAL_GB="$(awk -v k="${MEM_TOTAL_KB}" 'BEGIN{printf "%.1f", k/1024/1024}')"
MEM_AVAIL_GB="$(awk '/MemAvailable/{printf "%.1f", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo 0)"
info "总内存：${MEM_TOTAL_GB} GB"
info "可用内存：${MEM_AVAIL_GB} GB"

# 30B 级 MoE 的 NVFP4 权重约 17.5GB，加上 KV 缓存和运行时，24GB 是底线
if awk -v m="${MEM_TOTAL_GB}" 'BEGIN{exit !(m >= 26)}'; then
    ok "内存分配充足，可跑 Qwen3.6-35B-A3B 级别的 MoE"
elif awk -v m="${MEM_TOTAL_GB}" 'BEGIN{exit !(m >= 20)}'; then
    warn "内存 ${MEM_TOTAL_GB}GB，只能跑 Gemma-4-26B-A4B 或更小的模型"
else
    block "内存仅 ${MEM_TOTAL_GB}GB，跑不动 30B 级 MoE。请在 Windows 侧调整 .wslconfig 后执行 wsl --shutdown"
fi

# 磁盘：模型权重约 18GB，加依赖和 Python 环境，30GB 是底线
DISK_AVAIL_GB="$(df -BG / 2>/dev/null | awk 'NR==2{gsub("G","",$4); print $4}')"
if [ -n "${DISK_AVAIL_GB}" ]; then
    info "根分区可用：${DISK_AVAIL_GB} GB"
    if [ "${DISK_AVAIL_GB}" -ge 40 ] 2>/dev/null; then
        ok "磁盘余量充足"
    elif [ "${DISK_AVAIL_GB}" -ge 30 ] 2>/dev/null; then
        warn "磁盘可用 ${DISK_AVAIL_GB}GB，刚够用没有余量，建议先清理"
    else
        block "磁盘可用 ${DISK_AVAIL_GB}GB，放不下约 18GB 的模型权重加依赖。请清理空间或扩容虚拟磁盘"
    fi

    # WSL 专属提醒：根分区是动态扩展的虚拟磁盘，上面那个数字是容量上限，
    # 不代表宿主磁盘真的有那么多空间。ext4.vhdx 会随写入增长，最终撞到宿主磁盘的墙。
    if grep -qi microsoft /proc/version 2>/dev/null; then
        HOST_FREE="$(df -BG /mnt/c 2>/dev/null | awk 'NR==2{print $4}')"
        info ""
        info "注意：WSL 的根分区是动态扩展的虚拟磁盘，上面是「容量上限」而非真实可用量。"
        info "      ext4.vhdx 会随写入逐步增长，真正的天花板是宿主磁盘的剩余空间。"
        [ -n "${HOST_FREE}" ] && info "      当前宿主 C 盘可用：${HOST_FREE}（D 盘请另查）"
    fi
fi

# ================================================================ 步骤 4 安装 uv
step "步骤 4 / 5　安装 uv（Python 包管理器）"

if command -v uv >/dev/null 2>&1; then
    ok "uv 已安装：$(uv --version 2>&1)"
else
    info "正在安装 uv ..."
    if curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; then
        export PATH="${HOME}/.local/bin:${PATH}"
        if command -v uv >/dev/null 2>&1; then
            ok "uv 安装成功：$(uv --version 2>&1)"
        else
            block "uv 安装后仍不在 PATH 中，请手动执行：export PATH=\"\$HOME/.local/bin:\$PATH\""
        fi
    else
        block "uv 安装失败，请检查网络。手动安装：curl -LsSf https://astral.sh/uv/install.sh | sh"
    fi
fi

# ================================================================ 步骤 5 安装 FreeToken
step "步骤 5 / 5　安装 FreeToken"

if [ "${BLOCK}" -gt 0 ]; then
    warn "前面存在阻断项，跳过安装步骤。先解决阻断问题再重新运行本脚本"
else
    if [ ! -d "${VENV_DIR}" ]; then
        info "创建虚拟环境：${VENV_DIR}（Python ${PY_VERSION}）"
        if uv venv --python "${PY_VERSION}" "${VENV_DIR}" >/dev/null 2>&1; then
            ok "虚拟环境创建成功"
        else
            warn "指定 Python ${PY_VERSION} 失败，回退到系统默认版本"
            uv venv "${VENV_DIR}" >/dev/null 2>&1 && ok "虚拟环境创建成功（默认 Python 版本）"
        fi
    else
        ok "虚拟环境已存在：${VENV_DIR}"
    fi

    info "安装 freetoken[accel] ...（首次会下载若干 GB 依赖，通常 5～30 分钟）"
    info "这一步 uv 会实时打印下载进度，屏幕会动；只有卡住时才需要干预"
    # 不接管道：让 uv 直接写终端，才能显示下载进度条。
    # （旧版这里写了 | tail -n 3，tail 要等输入结束才输出，导致整个过程屏幕全黑。）
    if uv pip install --python "${VENV_DIR}/bin/python" "freetoken[accel]"; then
        ok "freetoken 安装完成"
    else
        block "freetoken 安装失败，请把上方 uv 打印的报错最后 20 行贴回给我"
    fi
fi

# ================================================================ nvcc 检查
step "附加检查　CUDA 工具链（nvcc）"

if command -v nvcc >/dev/null 2>&1; then
    NVCC_LINE="$(nvcc --version 2>&1 | grep -o 'release [0-9.]*' || echo '版本未知')"
    ok "nvcc 可用：${NVCC_LINE}"
else
    warn "PATH 上未找到 nvcc —— PyPI 版的内核需要 JIT 编译，没有 nvcc 无法运行"
    info ""
    info "两条路，推荐路线 A（只下 8MB，且免去首次 JIT 编译）："
    info ""
    info "  路线 A：装官方 nightly 预编译内核（cp312 构建，与本脚本所建 venv 一致）"
    info "    source ${VENV_DIR}/bin/activate"
    info "    uv pip install \\"
    info "      \"freetoken[accel] @ https://github.com/FlashML-org/FreeToken/releases/download/nightly/freetoken-0.1.3%2Bg0d652e73a-cp312-cp312-linux_x86_64.whl\" \\"
    info "      \"https://github.com/FlashML-org/FreeToken/releases/download/nightly/freetoken_kernel_cache-0.1.3%2Bcu130.g0d652e73a-py3-none-linux_x86_64.whl\""
    info "    说明：nightly 的 tag 每晚移动、旧轮子会被删掉，必须按上面的文件名钉住 URL，"
    info "          不能写成 tag 地址。轮子名里的 +g0d652e73a 是构建提交号。"
    info ""
    info "  路线 B：装 CUDA 13 toolkit（约 3GB，之后首次运行仍需 JIT 编译）"
    info "    wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb"
    info "    sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt update"
    info "    sudo apt install -y cuda-toolkit-13-0"
fi

# ================================================================ 汇总
step "环境准备汇总"

printf "  %d 项通过，%d 项警告，%d 项阻断\n" "${PASS}" "${WARN}" "${BLOCK}"

if [ "${BLOCK}" -gt 0 ]; then
    printf "\n  需要先解决的阻断项：\n%b\n" "${BLOCKERS}"
    printf "  解决后重新运行：bash %s\n\n" "$(basename "$0")"
    exit 1
fi

cat <<'EOF'

  全部就绪。接下有两条长活，分两个终端跑（B 慢，尽早开始）。

  ── 终端 A（先跑，快）── 内核 + 带宽校准
    source ~/freetoken-venv/bin/activate

    # A1) 预检 GitHub 直链（应返回 200；302 说明会跳转，也算通）
    curl -sIL -o /dev/null -w "HTTP %{http_code}\n" \
      "https://github.com/FlashML-org/FreeToken/releases/download/nightly/freetoken_kernel_cache-0.1.3%2Bcu130.g0d652e73a-py3-none-linux_x86_64.whl"

    # A2) 装官方 nightly 预编译内核（8MB，替代路线 B 那 3GB 的 CUDA toolkit）
    uv pip install \
      "freetoken[accel] @ https://github.com/FlashML-org/FreeToken/releases/download/nightly/freetoken-0.1.3%2Bg0d652e73a-cp312-cp312-linux_x86_64.whl" \
      "https://github.com/FlashML-org/FreeToken/releases/download/nightly/freetoken_kernel_cache-0.1.3%2Bcu130.g0d652e73a-py3-none-linux_x86_64.whl"
    ft --version          # 应带 +g0d652e73a 构建戳

    # A3) 带宽校准（每张显卡只需跑一次；决定 auto 从 offload 升级为 hybrid）
    ft bench bw --dtype nvfp4,bf16

  ── 终端 B（后开，慢，21.85GB）── 下载权重
    source ~/freetoken-venv/bin/activate
    uv pip install "huggingface_hub[cli]"

    # B1) 先测两个源，哪个通用哪个
    curl -sI -o /dev/null -m 10 -w "huggingface.co  %{http_code}  %{time_total}s\n" https://huggingface.co
    curl -sI -o /dev/null -m 10 -w "hf-mirror.com   %{http_code}  %{time_total}s\n" https://hf-mirror.com

    # B2) 官方不通时才需要挂镜像（国内多数家宽需挂）
    export HF_ENDPOINT=https://hf-mirror.com

    # B3) 下载（支持断点续传，断了重跑同一条命令即可）
    hf download nvidia/Qwen3.6-35B-A3B-NVFP4 --local-dir ~/models/Qwen3.6-35B-A3B-NVFP4

  ── 都完成后 —— 起服务（回到终端 A）
    ft serve --model ~/models/Qwen3.6-35B-A3B-NVFP4 \
      --text-model-only \
      --moe-strategy auto \
      --max-seq-len-override 16384 \
      --host 127.0.0.1 --port 1919 \
      --enable-cache-report

    # 另开一个终端验证与观测
    curl http://127.0.0.1:1919/v1/models
    ft ctl stats          # 吞吐、延迟、VRAM、缓存池占用
    ft shell              # 在终端里直接聊天

  内存预算与两个关键参数：
    权重 21.85GB 常驻主机内存（offload 策略下专家池全在 RAM）
    --memory-ratio 管的是「显存」（权重+MoE缓存+KV），不是主机内存，
    调它救不了 RAM 压力，别指望靠它解决装不下的问题。
    --text-model-only：Qwen3.6-35B-A3B 是多模态 checkpoint，不带这个参数
    会额外建视觉塔（白占一份显存与主机内存）；纯文本问答/ChatBI 用不上它。

  提示：
    - 16K 上下文是保守起步值，跑通后可逐步往上试
    - 若 --moe-strategy 最终解析为 offload（而非 hybrid），追加 --moe-cpu-layers auto
      （该参数专为 Windows/WSL 设计，因为 CUDA pinned memory 有上限）
    - Windows 侧应用可直接访问 http://127.0.0.1:1919（WSL2 自带 localhost 转发）
    - 服务运行期间别开浏览器、WPS 等占显存的程序
    - 权重务必留在虚拟磁盘内（~/models）；放 /mnt/d 走 9p 协议，慢到不可用

EOF
