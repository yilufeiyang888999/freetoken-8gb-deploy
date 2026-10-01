#!/usr/bin/env bash
# ==============================================================================
# FreeToken 权重导入 —— 把 Windows 侧迅雷下好的权重搬进 WSL
# ------------------------------------------------------------------------------
# 背景：
#   21.85 GB 的权重用迅雷在 Windows 上拿下来了，现在要进 WSL 的 ext4。
#
# 为什么不直接让 ft serve 读 /mnt/d：
#   /mnt/d 走的是 9p 文件系统，随机读延迟比 ext4 高一个量级。
#   safetensors 加载时要做大量 mmap/随机读，从 9p 直接加载不仅慢，
#   还会让首次启动卡到你以为死机。所以必须先复制进 ext4。
#
# 注意：本脚本注释中文，但变量名/函数名必须 ASCII
#       （bash 只认 [a-zA-Z_][a-zA-Z0-9_]*，中文变量名会报语法错误）。
#
# 用法：
#   bash import_model_from_windows.sh /mnt/d/models/Qwen3.6-35B-A3B-NVFP4
#   bash import_model_from_windows.sh <源目录> --fresh        # 先清空目标目录再复制
#   bash import_model_from_windows.sh <源目录> --clean-src    # 校验通过后删掉源
#   bash import_model_from_windows.sh --verify                # 只校验目标目录
#
#   也可用 FT_MODEL_DIR=<路径> 覆盖目标位置（默认 ~/models/Qwen3.6-35B-A3B-NVFP4）
# ==============================================================================

set -uo pipefail

# 可用环境变量覆盖目标路径（便于测试，也便于你把模型放到别的盘）
MODEL_DIR="${FT_MODEL_DIR:-${HOME}/models/Qwen3.6-35B-A3B-NVFP4}"

# 文件名:字节数（与 download_freetoken_model.sh 保持一致）
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
EXPECT_BYTES=23462477857   # 21.85 GB，与上面 17 个字节数之和一致（已逐项核算）

step() { printf "\n%s\n %s\n%s\n" "────────────────────────────────────────────────────────────────" "$1" "────────────────────────────────────────────────────────────────"; }
ok()   { printf "  [通过] %s\n" "$1"; }
warn() { printf "  [警告] %s\n" "$1"; }
bad()  { printf "  [阻断] %s\n" "$1"; }
info() { printf "  %s\n" "$1"; }

# ================================================================ 校验
run_verify() {
    local fail=0 got want name pair path
    step "完整性校验　${MODEL_DIR}"
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
        bad "校验未通过 —— 不要拿去启动服务"
        return 1
    fi
    ok "全部 ${EXPECT_FILES} 个文件字节数一致，权重完整"
    return 0
}

# ================================================================ 参数解析
SRC=""
CLEAN_SRC=0
FRESH=0
VERIFY_ONLY=0
for arg in "$@"; do
    case "${arg}" in
        --clean-src) CLEAN_SRC=1 ;;
        --fresh)     FRESH=1 ;;
        --verify)    VERIFY_ONLY=1 ;;
        -*)          bad "未知参数 ${arg}"; exit 2 ;;
        *)           SRC="${arg}" ;;
    esac
done

printf "\n%s\n FreeToken 权重导入　Windows -> WSL\n%s\n" \
  "════════════════════════════════════════════════════════════════" \
  "════════════════════════════════════════════════════════════════"

if [ "${VERIFY_ONLY}" -eq 1 ]; then
    if run_verify; then exit 0; else exit 1; fi
fi

if [ -z "${SRC}" ]; then
    bad "请给出 Windows 侧源目录，例如："
    info "  bash $0 /mnt/d/models/Qwen3.6-35B-A3B-NVFP4"
    info ""
    info "不知道盘符挂在哪： ls /mnt"
    exit 2
fi

if [ ! -d "${SRC}" ]; then
    bad "源目录不存在：${SRC}"
    info "先确认： ls /mnt ; 然后 ls ${SRC}"
    exit 2
fi

# ================================================================ 步骤 1 源检查
step "步骤 1 / 4　源目录检查"
info "源：${SRC}"

SRC_GB="$(du -sb "${SRC}" 2>/dev/null | awk '{printf "%.2f", $1/1024/1024/1024}')"
info "源大小：${SRC_GB} GB（期望 21.85 GB）"

SRC_MISSING=0
for pair in "${FILES[@]}"; do
    name="${pair%%:*}"
    [ -f "${SRC}/${name}" ] || { bad "源里缺 ${name}"; SRC_MISSING=1; }
done
# 只有明确报告过缺失才继续说；顺带兜住「源里文件名没整理过」的情况
if [ "${SRC_MISSING}" -ne 0 ]; then
    info ""
    info "若文件都在、只是文件名是乱码/哈希名，先在 Windows 上跑整理脚本："
    info "  python collect_model_files.py <迅雷下载目录> D:\\models\\Qwen3.6-35B-A3B-NVFP4"
    exit 1
fi
ok "17 个文件全部就位"

# ================================================================ 步骤 2 空间
step "步骤 2 / 4　目标空间检查"

# 先量目标目录现占多少 —— 必须赶在 mkdir 之前量，
# 否则新建出来的空目录恒为 0，--fresh 的提示就成了废话。
OLD_BYTES="$(du -sb "${MODEL_DIR}" 2>/dev/null | awk '{print $1}')"
OLD_GB="$(awk -v b="${OLD_BYTES:-0}" 'BEGIN{printf "%.2f", b/1024/1024/1024}')"

# 必须创建 MODEL_DIR 本身（不只是父目录）—— 少了这句，第一次导入时
# 目标目录不存在，后面每个 cp 都会以 "No such file or directory" 失败。
mkdir -p "${MODEL_DIR}"

# 目标必须和家目录同一个分区（都要在 ext4 上）
AVAIL_KB="$(df -Pk "${HOME}" | awk 'NR==2{print $4}')"
AVAIL_GB="$(awk -v k="${AVAIL_KB:-0}" 'BEGIN{printf "%.1f", k/1024/1024}')"
info "WSL ext4 可用：${AVAIL_GB} GB（需要约 22 GB）"

if [ "${OLD_GB}" != "0.00" ]; then
    info "目标目录已有 ${OLD_GB} GB 内容，同名文件将被覆盖"
fi

if awk -v g="${AVAIL_GB}" 'BEGIN{exit !(g >= 25)}'; then
    ok "空间充足"
elif awk -v g="${AVAIL_GB}" 'BEGIN{exit !(g >= 22)}'; then
    warn "空间刚够，装完没有余量"
else
    bad "空间不足 ${AVAIL_GB}GB"
    info "WSL 根分区显示的容量上限是虚拟磁盘的，真正的天花板是宿主 D 盘。"
    info "清理后重跑本脚本；已复制的文件会跳过。"
    exit 1
fi

# ================================================================ 步骤 3 复制
step "步骤 3 / 4　复制到 ext4"

# --fresh：清空目标目录再复制。用于清理之前中断下载留下的残片。
# 只删脚本自己管的模型目录，路径写死在 MODEL_DIR，不会有别的歧义。
if [ "${FRESH}" -eq 1 ]; then
    if [ -d "${MODEL_DIR}" ]; then
        info "清空目标目录（--fresh）：${MODEL_DIR}"
        info "  将删除 ${OLD_GB} GB，含之前中断下载留下的残片"
        if rm -rf -- "${MODEL_DIR}"; then
            ok "已清空"
        else
            bad "清空失败，检查权限：ls -ld ${MODEL_DIR}"
            exit 1
        fi
    fi
    mkdir -p "${MODEL_DIR}"
fi

info "从 9p 挂载点复制 21.85 GB，通常 1~5 分钟，期间看不到进度是正常的。"
info ""

COPIED=0
SKIPPED=0
FAILED=0
T0="$(date +%s)"

for pair in "${FILES[@]}"; do
    name="${pair%%:*}"
    want="${pair##*:}"
    dst="${MODEL_DIR}/${name}"
    src="${SRC}/${name}"

    if [ -f "${dst}" ] && [ "$(stat -c '%s' "${dst}")" = "${want}" ]; then
        info "跳过（已完整）  ${name}"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    printf "  --> %s ... " "${name}"
    if cp -f "${src}" "${dst}"; then
        got="$(stat -c '%s' "${dst}")"
        if [ "${got}" = "${want}" ]; then
            printf "完成\n"
            COPIED=$((COPIED + 1))
        else
            printf "字节数不符（%s / %s）\n" "${got}" "${want}"
            FAILED=$((FAILED + 1))
        fi
    else
        printf "复制失败\n"
        FAILED=$((FAILED + 1))
    fi
done

T1="$(date +%s)"
ELAPSED=$((T1 - T0))
info ""
info "耗时 ${ELAPSED} 秒，新复制 ${COPIED} 个，跳过 ${SKIPPED} 个，失败 ${FAILED} 个"

if [ "${FAILED}" -ne 0 ]; then
    bad "有文件复制失败，重跑本脚本即可（已完整的会自动跳过）"
    exit 1
fi

# ================================================================ 步骤 4 校验
if ! run_verify; then exit 1; fi

# ================================================================ 可选清理
if [ "${CLEAN_SRC}" -eq 1 ]; then
    step "清理 Windows 侧副本"
    info "将删除：${SRC}"
    info "释放约 ${SRC_GB} GB（WSL 侧已校验通过，此副本不再需要）"
    if rm -rf -- "${SRC}"; then
        ok "已删除"
    else
        warn "删除失败，可手动处理"
    fi
fi

# ================================================================ 完成
step "全部就绪"
cat <<'EOF'
  启动服务

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
    - --text-model-only 必加：多模态 checkpoint，不加会额外建视觉塔，
      白占一份显存与内存；纯文本问答 / ChatBI 用不上。
    - 启动前把浏览器、WPS、微信关掉 —— 8GB 显存是这条链路上最紧的一环。
    - ft bench bw 已确认 backend=hybrid（CPU/PCIe 5.03x），
      CPU/GPU 并行路径会生效，速度应优于纯 offload。

EOF
exit 0
