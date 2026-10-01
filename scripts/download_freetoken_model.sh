#!/usr/bin/env bash
# ==============================================================================
# FreeToken 模型权重下载 —— nvidia/Qwen3.6-35B-A3B-NVFP4
# ------------------------------------------------------------------------------
# 用途：自动选源 + 三级降级下载 + 分片字节级完整性校验。
#
# 为什么需要三级降级（都是实测踩出来的）：
#   1) 本机 Windows hosts 把 huggingface.co 指向 127.0.0.1，
#      WSL 通过 DNS 转发继承劫持 → 官方源必然失败，必须走 hf-mirror。
#   2) 新版 huggingface_hub 默认走 Xet 存储后端，而 Xet 的 CAS 服务器
#      (cas-server.xethub.hf.co) 不受 HF_ENDPOINT 控制 → 挂镜像后报
#      401 Unauthorized。必须 HF_HUB_DISABLE_XET=1 才会退回普通 HTTPS。
#   3) 若 huggingface_hub 仍然出问题，最后用 curl 直接打镜像的
#      resolve 直链，完全绕开这个库。
#
# 安全：仓库已核实 gated=False / private=False，无需 token。
#       校验用「每个文件的确切字节数」（取自仓库 API，非估算）。
#
# 注意：本脚本注释用中文，但变量名/函数名必须 ASCII ——
#       bash 的标识符只允许 [a-zA-Z_][a-zA-Z0-9_]*，中文变量名会报语法错误。
#       （Python 支持中文变量名，bash 不支持。）
#
# 用法：
#   bash download_freetoken_model.sh           # 前台，自动三级降级
#   bash download_freetoken_model.sh bg        # 后台下载，日志 ~/models/download.log
#   bash download_freetoken_model.sh curl      # 直接走 curl 直链通道
#   bash download_freetoken_model.sh verify    # 只做完整性校验（已下完时用）
# ==============================================================================

set -uo pipefail

MODEL_REPO="nvidia/Qwen3.6-35B-A3B-NVFP4"
MODEL_REV="1355db6a052410cfd62085d94b58866fd0f2c3c5"   # 钉住版本，防止上游中途变动
MODEL_DIR="${HOME}/models/Qwen3.6-35B-A3B-NVFP4"
LOG_FILE="${HOME}/models/download.log"
MIRROR="https://hf-mirror.com"
VENV_DIR="${HOME}/freetoken-venv"

MODE="${1:-}"

# 完整文件清单（名称:字节数）—— 取自仓库 API 实测，非估算。
# 既用于 curl 通道下载，也用于最终校验，让脚本自包含、不依赖运行时查 API。
FILES=(
  "model-00001-of-00003.safetensors:10006877608"
  "model-00002-of-00003.safetensors:10003595752"
  "model-00003-of-00003.safetensors:3413864960"
  "model.safetensors.index.json:13726227"
  "tokenizer.json:12807982"
  "vocab.json:6722759"
  ".quant_summary.txt:4752286"
  "config.json:58110"
  "hf_quant_config.json:35085"
  "tokenizer_config.json:16718"
  "README.md:9936"
  "chat_template.jinja:7764"
  ".gitattributes:1635"
  "preprocessor_config.json:390"
  "video_preprocessor_config.json:385"
  "generation_config.json:202"
  "configuration.json:58"
)
EXPECT_FILES="${#FILES[@]}"

step() { printf "\n%s\n %s\n%s\n" "────────────────────────────────────────────────────────────────" "$1" "────────────────────────────────────────────────────────────────"; }
ok()   { printf "  [通过] %s\n" "$1"; }
warn() { printf "  [警告] %s\n" "$1"; }
bad()  { printf "  [阻断] %s\n" "$1"; }
info() { printf "  %s\n" "$1"; }

if ! command -v uv >/dev/null 2>&1 && [ -x "${HOME}/.local/bin/uv" ]; then
    export PATH="${HOME}/.local/bin:${PATH}"
fi

printf "\n%s\n FreeToken 权重下载：%s\n%s\n" \
  "════════════════════════════════════════════════════════════════" \
  "${MODEL_REPO}" \
  "════════════════════════════════════════════════════════════════"

# ================================================================ 校验函数
# 用「每个文件的确切字节数」判定完整性：大文件差一个字节都说明没下完。
run_verify() {
    local fail=0 got want name path pair
    step "完整性校验"

    for pair in "${FILES[@]}"; do
        name="${pair%%:*}"
        want="${pair##*:}"
        path="${MODEL_DIR}/${name}"
        if [ ! -f "${path}" ]; then
            bad "缺失 ${name}"
            fail=1
            continue
        fi
        got="$(stat -c '%s' "${path}" 2>/dev/null || echo 0)"
        if [ "${got}" = "${want}" ]; then
            ok "${name}"
        else
            bad "${name} 不完整：实际 ${got} / 期望 ${want} 字节"
            fail=1
        fi
    done

    if [ "${fail}" -ne 0 ]; then
        bad "校验未通过 —— 不要拿去启动服务；重跑本脚本会从断点续传"
        return 1
    fi
    ok "全部 ${EXPECT_FILES} 个文件字节数一致，权重完整"
    return 0
}

# ================================================================ verify 模式
if [ "${MODE}" = "verify" ]; then
    if run_verify; then exit 0; else exit 1; fi
fi

# ================================================================ 步骤 1 选源
step "步骤 1 / 5　自动选源"

probe() {
    local code
    code="$(curl -sIL -o /dev/null -m 10 -w '%{http_code}' "$1" 2>/dev/null)"
    [ -z "${code}" ] && code="-1"
    echo "${code}"
}

HF_CODE="$(probe https://huggingface.co/api/models/${MODEL_REPO})"
MI_CODE="$(probe ${MIRROR}/api/models/${MODEL_REPO})"
info "huggingface.co  ->  HTTP ${HF_CODE}"
info "hf-mirror.com   ->  HTTP ${MI_CODE}"

if [ "${HF_CODE}" = "200" ]; then
    # 注意必须 unset 而不是置空：HF_ENDPOINT="" 会被 huggingface_hub
    # 当成有效值用起来，拼出 "/api/..." 这种残缺 URL。
    unset HF_ENDPOINT
    export HF_HUB_DISABLE_XET=1
    ok "官方源可用，使用官方源（已禁用 Xet）"
elif [ "${MI_CODE}" = "200" ]; then
    export HF_ENDPOINT="${MIRROR}"
    # 关键：镜像不代理 Xet 的 CAS 服务器，不关掉必定 401
    export HF_HUB_DISABLE_XET=1
    ok "官方源不可用，改用镜像 ${MIRROR}（已禁用 Xet）"
else
    bad "两个源都不可达"
    info "查：getent ahosts huggingface.co   （若为 127.0.0.1，是 Windows hosts 劫持）"
    exit 1
fi

# 大分片最终会 302 到 Xet 的 CAS bridge 取数据（镜像只代理元数据）。
# 这个域名不受 HF_ENDPOINT 控制，单独探一下，失败也在这一步就暴露，
# 不用等到下载 9GB 时才发现。
CB_CODE="$(probe https://cas-bridge.xethub.hf.co/)"
if [ "${CB_CODE}" = "-1" ]; then
    warn "cas-bridge.xethub.hf.co 不可达（HTTP ${CB_CODE}）"
    info "  大分片(.safetensors)的数据实际由这个域名提供，不通会导致大文件下载失败。"
    info "  小文件仍能下 —— 脚本会照常继续，若卡在大分片请把日志贴回给我。"
else
    ok "Xet CAS bridge 可达（HTTP ${CB_CODE}），大分片下载路径通"
fi

# ================================================================ 步骤 2 空间
step "步骤 2 / 5　磁盘空间"

AVAIL_KB="$(df -Pk "${HOME}" 2>/dev/null | awk 'NR==2{print $4}')"
AVAIL_GB="$(awk -v k="${AVAIL_KB:-0}" 'BEGIN{printf "%.1f", k/1024/1024}')"
info "家目录可用：${AVAIL_GB} GB（需要约 22 GB）"
if awk -v g="${AVAIL_GB}" 'BEGIN{exit !(g >= 25)}'; then
    ok "空间充足"
elif awk -v g="${AVAIL_GB}" 'BEGIN{exit !(g >= 22)}'; then
    warn "空间刚够，装完没有余量，建议先清理"
else
    bad "空间不足 ${AVAIL_GB}GB"
    info "提示：WSL 根分区那个「947GB 可用」是虚拟磁盘容量上限，"
    info "      真正的天花板是宿主 D 盘的剩余空间（ext4.vhdx 随写入增长）。"
    exit 1
fi

# ================================================================ 步骤 3 工具
step "步骤 3 / 5　准备下载工具"

if [ ! -x "${VENV_DIR}/bin/python" ]; then
    bad "未找到虚拟环境 ${VENV_DIR}，请先运行 wsl_setup_freetoken.sh"
    exit 1
fi

info "安装 huggingface_hub[cli] ..."
if uv pip install --python "${VENV_DIR}/bin/python" "huggingface_hub[cli]"; then
    ok "huggingface_hub 就绪"
else
    bad "huggingface_hub 安装失败"
    exit 1
fi

USE_TRANSFER=0
if uv pip install --python "${VENV_DIR}/bin/python" hf_transfer >/dev/null 2>&1; then
    USE_TRANSFER=1
    ok "hf_transfer 就绪"
else
    warn "hf_transfer 未装上，走普通 HTTP"
fi

HF_BIN="${VENV_DIR}/bin/hf"
[ -x "${HF_BIN}" ] || HF_BIN="${VENV_DIR}/bin/huggingface-cli"
[ -x "${HF_BIN}" ] || { bad "虚拟环境里找不到 hf 命令"; exit 1; }
info "下载命令：${HF_BIN}"

mkdir -p "${MODEL_DIR}"

# ================================================================ curl 通道
# 完全绕开 huggingface_hub：直接打镜像的 resolve 直链。
#
# 续传行为已实测（重要）：
#   镜像先 307/302 重定向。落点分两种：
#     - 大分片(.safetensors) → 跳到 cas-bridge.xethub.hf.co，支持 Range，
#       返回 206，断点续传真实有效；
#     - 小文件(config.json 等) → 走镜像自身缓存，返回 200，忽略 Range，
#       curl -C - 会以 exit 33 (CURLE_RANGE_ERROR) 中止。
#   所以这里先试续传，失败就整体重下 —— 小文件重下代价可忽略。
run_curl_channel() {
    local ep="${HF_ENDPOINT:-https://huggingface.co}"
    local pair name url sub target rc
    step "curl 直链通道"
    info "源：${ep}"
    info "大文件支持断点续传；小文件若不能续传会自动整体重下。"
    info ""

    for pair in "${FILES[@]}"; do
        name="${pair%%:*}"
        url="${ep}/${MODEL_REPO}/resolve/${MODEL_REV}/${name}"
        sub="$(dirname "${name}")"
        [ "${sub}" != "." ] && mkdir -p "${MODEL_DIR}/${sub}"
        target="${MODEL_DIR}/${name}"

        printf "  --> %s\n" "${name}"

        # 已有部分内容 → 先试续传
        if [ -s "${target}" ]; then
            if curl -fL -C - --retry 3 --retry-delay 3 --connect-timeout 15 \
                     --speed-limit 1024 --speed-time 180 \
                     --progress-bar -o "${target}" "${url}"; then
                info "    续传完成"
                continue
            fi
            rc=$?
            warn "    续传不适用（curl rc=${rc}），改为整体重下"
        fi

        # 全新下载 / 整体重下。不加 --max-time：
        # 9GB 级文件下载可能耗时很久，靠 --speed-limit 检测停滞即可。
        if curl -fL --retry 3 --retry-delay 3 --connect-timeout 15 \
                 --speed-limit 1024 --speed-time 180 \
                 --progress-bar -o "${target}" "${url}"; then
            info "    完成"
        else
            bad "下载失败：${name}"
            info "重跑本命令即可从该文件继续（已完成的大文件会续传）"
            return 1
        fi
    done
    ok "curl 通道下载完成"
    return 0
}

# ================================================================ 步骤 4 下载
step "步骤 4 / 5　下载权重（21.85 GB）"
info "目标目录：${MODEL_DIR}"
info "版本：${MODEL_REV}"

# --- 后台模式：把「同一个脚本 + curl 模式」丢到 nohup，保持行为一致 ---
if [ "${MODE}" = "bg" ]; then
    info "后台下载模式，日志：${LOG_FILE}"
    nohup bash "$0" curl > "${LOG_FILE}" 2>&1 &
    ok "已在后台启动，PID $!"
    info ""
    info "看体积（比看日志清爽，推荐）："
    info "  watch -n 10 'du -sh ${MODEL_DIR}'"
    info "看日志尾部："
    info "  tail -c 2000 ${LOG_FILE}"
    info "下完后校验：bash $0 verify"
    exit 0
fi

if [ "${MODE}" = "curl" ]; then
    if run_curl_channel && run_verify; then exit 0; fi
    exit 1
fi

# --- 默认：三级降级 ---
info "策略：hf + hf_transfer  ->  hf 普通 HTTP  ->  curl 直链"
info "中断不要紧，重跑同一条命令会自动续传。"
info ""

DL_OK=0

# 尝试 1：hf + hf_transfer
if [ "${USE_TRANSFER}" -eq 1 ]; then
    info "[尝试 1/3] hf download + hf_transfer"
    if HF_HUB_ENABLE_HF_TRANSFER=1 "${HF_BIN}" download "${MODEL_REPO}" \
            --revision "${MODEL_REV}" --local-dir "${MODEL_DIR}"; then
        DL_OK=1
    else
        warn "尝试 1 失败"
    fi
fi

# 尝试 2：hf 普通 HTTP
if [ "${DL_OK}" -eq 0 ]; then
    info "[尝试 2/3] hf download（普通 HTTP，不带 hf_transfer）"
    if HF_HUB_ENABLE_HF_TRANSFER=0 "${HF_BIN}" download "${MODEL_REPO}" \
            --revision "${MODEL_REV}" --local-dir "${MODEL_DIR}"; then
        DL_OK=1
    else
        warn "尝试 2 失败"
    fi
fi

# 尝试 3：curl 直链，完全绕开 huggingface_hub
if [ "${DL_OK}" -eq 0 ]; then
    info "[尝试 3/3] curl 直链通道"
    if run_curl_channel; then
        DL_OK=1
    fi
fi

if [ "${DL_OK}" -eq 0 ]; then
    bad "三条通道全部失败"
    info ""
    info "把上面最后一段报错原文贴回给我。另外可自查："
    info "  curl -sI ${MIRROR}/${MODEL_REPO}/resolve/main/config.json | head -3"
    exit 1
fi

# ================================================================ 步骤 5 校验
if run_verify; then
    step "全部就绪"
    cat <<'EOF'
  下一步：启动服务

    source ~/freetoken-venv/bin/activate

    ft serve --model ~/models/Qwen3.6-35B-A3B-NVFP4 \
      --text-model-only \
      --moe-strategy auto \
      --max-seq-len-override 16384 \
      --host 127.0.0.1 --port 1919 \
      --enable-cache-report

  另开一个 WSL 终端：
    curl http://127.0.0.1:1919/v1/models
    ft ctl stats        # 吞吐、延迟、VRAM、缓存池占用
    ft shell            # 终端里直接聊天

  提示：
    - --text-model-only 必加：多模态 checkpoint，不带会额外建视觉塔，
      白占一份显存与内存；纯文本问答 / ChatBI 用不上。
    - 启动前把浏览器、WPS、微信关掉 —— 8GB 显存是这条链路上最紧的一环。
    - ft bench bw 已确认 backend=hybrid（CPU/PCIe 比值 5.03x），
      这条 CPU/GPU 并行路径会生效，速度应优于纯 offload。

EOF
    exit 0
else
    exit 1
fi
