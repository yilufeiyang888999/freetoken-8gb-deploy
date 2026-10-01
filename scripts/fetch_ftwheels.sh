#!/usr/bin/env bash
# ==============================================================================
# FreeToken nightly 预编译内核 —— 下载 + 校验 + 安装
# ------------------------------------------------------------------------------
# 用途：绕过 uv 直连 GitHub 的失败（Connection refused），用多通道降级下载
#       nightly 的两个轮子，校验官方 sha256 后装入 ~/freetoken-venv。
#
# 为什么需要它：
#   PyPI 版 freetoken 的 CUDA 内核要 JIT 编译，必须有 nvcc；
#   官方 nightly 提供预编译内核轮子（两个加起来约 8MB），免 nvcc、免 JIT。
#   没有内核时 ft bench bw 的 PCIe-gather 测不出来，--moe-strategy auto
#   只能退到 offload，拿不到 hybrid 的 CPU/PCIe 并行加速。
#
# 安全：经第三方加速通道下载的轮子必须校验哈希，对不上绝不安装。
#
# 注意：本脚本注释用中文，但**变量名必须用 ASCII** —— bash 的标识符只允许
#       [a-zA-Z_][a-zA-Z0-9_]*，中文变量名会被当成命令名而报语法错误。
#       （Python 支持中文变量名，bash 不支持，这点和 Python 不一样。）
#
# 用法：bash fetch_ftwheels.sh
# ==============================================================================

set -uo pipefail

VENV_DIR="${HOME}/freetoken-venv"
WORK_DIR="${HOME}/ftwheels"

BASE="https://github.com/FlashML-org/FreeToken/releases/download/nightly"
RT_FILE="freetoken-0.1.3+g0d652e73a-cp312-cp312-linux_x86_64.whl"
KC_FILE="freetoken_kernel_cache-0.1.3+cu130.g0d652e73a-py3-none-linux_x86_64.whl"
RT_URL="${BASE}/freetoken-0.1.3%2Bg0d652e73a-cp312-cp312-linux_x86_64.whl"
KC_URL="${BASE}/freetoken_kernel_cache-0.1.3%2Bcu130.g0d652e73a-py3-none-linux_x86_64.whl"

# 官方 release 页公布的哈希
RT_SHA="1f84dba5787f5aa212fd81ef3629cb1cf5ba9f911971f9f63a6fe603a5b882bb"
KC_SHA="aabc2f8d8a2dd6b9a5518a0b50b8e708dec7316bd064bd2f7efa0cc09ca3892c"

step() { printf "\n%s\n %s\n%s\n" "────────────────────────────────────────────────────────────────" "$1" "────────────────────────────────────────────────────────────────"; }
ok()   { printf "  [通过] %s\n" "$1"; }
warn() { printf "  [警告] %s\n" "$1"; }
bad()  { printf "  [阻断] %s\n" "$1"; }
info() { printf "  %s\n" "$1"; }

# uv 通常装在 ~/.local/bin，新开的 shell 可能不在 PATH 上
if ! command -v uv >/dev/null 2>&1; then
    if [ -x "${HOME}/.local/bin/uv" ]; then
        export PATH="${HOME}/.local/bin:${PATH}"
    fi
fi
if ! command -v uv >/dev/null 2>&1; then
    bad "PATH 上找不到 uv。请先执行 source \$HOME/.local/bin/env 或重新打开终端"
    exit 1
fi

printf "\n%s\n FreeToken nightly 内核安装\n%s\n" \
  "════════════════════════════════════════════════════════════════" \
  "════════════════════════════════════════════════════════════════"

# ================================================================ 步骤 1 网络预检
step "步骤 1 / 5　网络预检"

# 关键认知：WSL 的 DNS 是转发给 Windows 的，而 Windows 解析器会先读它自己的 hosts 文件。
# 所以只要 Windows 侧把 github.com 指到 127.0.0.1，WSL 里也会解析到 127.0.0.1，
# 表现是连接被立刻拒绝（Connection refused），而不是超时。
GH_IP="$(getent ahosts github.com 2>/dev/null | awk 'NR==1{print $1}')"
if [ -z "${GH_IP}" ]; then
    warn "无法解析 github.com"
elif [ "${GH_IP}" = "127.0.0.1" ] || [ "${GH_IP}" = "0.0.0.0" ]; then
    warn "github.com 解析到 ${GH_IP} —— 源头在 Windows 的 hosts 文件"
    info "  WSL 自己的 /etc/hosts 一般是干净的，别在那里白找。真正的位置："
    info "  C:\\Windows\\System32\\drivers\\etc\\hosts"
    info "  本脚本走加速通道可以绕开，照常继续；"
    info "  但 git clone / pip 装 GitHub 包仍会失败，建议另行清理。"
else
    ok "github.com 解析正常：${GH_IP}"
fi

# 模型权重走 HuggingFace。huggingface.co 若同样被 hosts 指向 127.0.0.1，
# 下载 21.85GB 权重时必须改用国内镜像。
HF_IP="$(getent ahosts huggingface.co 2>/dev/null | awk 'NR==1{print $1}')"
if [ "${HF_IP}" = "127.0.0.1" ] || [ "${HF_IP}" = "0.0.0.0" ]; then
    warn "huggingface.co 也解析到 ${HF_IP} —— 下权重时请设 HF_ENDPOINT=https://hf-mirror.com"
else
    ok "huggingface.co 解析：${HF_IP:-未知}"
fi

mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}" || exit 1

# ================================================================ 步骤 2 下载
step "步骤 2 / 5　下载轮子（多通道自动降级）"
info "工作目录：${WORK_DIR}"

# 加速通道优先（实测 gh-proxy.com / ghproxy.net 可用）；
# 直连放后面兜底 —— 只要 Windows hosts 里有那条 github 记录，直连必然被拒。
CHANNELS=("https://gh-proxy.com/" "https://ghproxy.net/" "" "https://ghfast.top/")
DL_DONE=""
for PREFIX in "${CHANNELS[@]}"; do
    CH_NAME="${PREFIX:-直连}"
    info "尝试：${CH_NAME}"
    if curl -fL --retry 2 --retry-delay 2 --connect-timeout 10 --max-time 300 \
             -o "${RT_FILE}" "${PREFIX}${RT_URL}" 2>/dev/null \
       && curl -fL --retry 2 --retry-delay 2 --connect-timeout 10 --max-time 300 \
             -o "${KC_FILE}" "${PREFIX}${KC_URL}" 2>/dev/null; then
        ok "下载成功：${CH_NAME}"
        DL_DONE=1
        break
    fi
    warn "${CH_NAME} 失败，换下一个"
done

if [ -z "${DL_DONE}" ]; then
    bad "所有通道都失败"
    info ""
    info "常见原因（按可能性排序）："
    info "  1) Windows 的 hosts 文件把 github.com 指向 127.0.0.1"
    info "     ← 这个最常见。WSL 的 DNS 转发给 Windows，会继承这个改写。"
    info "     查（在 WSL 里）：getent ahosts github.com"
    info "     查（在 Windows 里，看真实源头）："
    info "       PowerShell: Get-Content C:\\Windows\\System32\\drivers\\etc\\hosts | Select-String github"
    info "     处理：确认是哪个工具写的，先关掉那个工具，再清理这批 127.0.0.1 条目。"
    info "  2) 代理变量指向 WSL 里的死端口（代理其实跑在 Windows 上）"
    info "     查：env | grep -i proxy            解：unset，或改用宿主 IP"
    info "         （宿主 IP 取 ip route show default 里的网关地址）"
    info "  3) 加速通道临时不可用 —— 换 https://ghproxy.net/ 或后面几个再试"
    info ""
    info "也可以手动把两个轮子下载后放进 ${WORK_DIR}，再重跑本脚本（会自动跳过已下好的）。"
    exit 1
fi

# ================================================================ 步骤 3 校验
step "步骤 3 / 5　校验哈希（经第三方通道下载时必做）"

SHA_FAIL=0
if echo "${RT_SHA}  ${RT_FILE}" | sha256sum -c - >/dev/null 2>&1; then
    ok "运行时轮子哈希一致"
else
    bad "运行时轮子哈希不匹配"
    SHA_FAIL=1
fi
if echo "${KC_SHA}  ${KC_FILE}" | sha256sum -c - >/dev/null 2>&1; then
    ok "内核轮子哈希一致"
else
    bad "内核轮子哈希不匹配"
    SHA_FAIL=1
fi

info "实际哈希："
sha256sum ./*.whl 2>/dev/null | sed 's/^/    /'

if [ "${SHA_FAIL}" -ne 0 ]; then
    bad "哈希校验未通过，已拒绝安装 —— 文件可能在中间环节被替换过"
    info "请删除 ${WORK_DIR} 下的文件后更换通道重试"
    exit 1
fi

# ================================================================ 步骤 4 安装
step "步骤 4 / 5　安装到虚拟环境"

if [ ! -x "${VENV_DIR}/bin/python" ]; then
    bad "未找到虚拟环境 ${VENV_DIR}"
    info "请先运行 wsl_setup_freetoken.sh 创建环境"
    exit 1
fi
info "目标环境：${VENV_DIR}"

# 注意：必须保留 [accel] 这个 extra，直接写 ./xxx.whl 会把它丢掉
if uv pip install --python "${VENV_DIR}/bin/python" \
        "freetoken[accel] @ file://${WORK_DIR}/${RT_FILE}" \
        "${WORK_DIR}/${KC_FILE}"; then
    ok "安装命令执行完成"
else
    bad "安装失败，请把上方报错贴回给我"
    exit 1
fi

# ================================================================ 步骤 5 验证
step "步骤 5 / 5　验证"

FT_VERSION="$("${VENV_DIR}/bin/ft" --version 2>&1)"
info "ft --version ：${FT_VERSION}"
case "${FT_VERSION}" in
    *g0d652e73a*)
        ok "已带 +g0d652e73a 构建戳，nightly 版本生效"
        ;;
    *)
        warn "版本里没看到构建戳，可能仍是 PyPI 版，请核对上面的输出"
        ;;
esac

cat <<'EOF'

  下一步：
    source ~/freetoken-venv/bin/activate
    ft bench bw --dtype nvfp4,bf16

  期望看到的变化：
    上一次 PCIe-gather 那列是 n/a、backend 是 offload；
    本内核包内含 freetoken__fast_index_copy_* 与 freetoken__batch_memcpy 等
    82 个预编译 .so，正是 gather 路径用的内核，装后应能测出 gather 数值，
    backend 有机会升级为 hybrid。

  若仍报 CUDA_HOME 缺失 —— 说明这个内核包不覆盖该 benchmark 用到的 kernel，
  改走 apt 装 CUDA 13 toolkit（约 3GB）：
    wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt update
    sudo apt install -y cuda-toolkit-13-0

  下载模型权重（21.85GB）——若 huggingface.co 被 hosts 拦，必须挂镜像：
    export HF_ENDPOINT=https://hf-mirror.com
    uv pip install "huggingface_hub[cli]"
    hf download nvidia/Qwen3.6-35B-A3B-NVFP4 --local-dir ~/models/Qwen3.6-35B-A3B-NVFP4

EOF
