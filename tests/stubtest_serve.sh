#!/usr/bin/env bash
# 用假 ft / 假 python / 假 site-packages 验证启动脚本的分支逻辑（沙箱内运行）
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="${REPO}/scripts/serve_freetoken.sh"
T="${REPO}/tests/.stub-tmp"
SP="$T/home/freetoken-venv/lib/python3.12/site-packages"
rm -rf "$T"
mkdir -p "$T/home/freetoken-venv/bin" "$T/home/models/Qwen3.6-35B-A3B-NVFP4" "$T/bin" "$SP/freetoken/server" "$SP/freetoken/attention"
for i in $(seq 1 17); do : > "$T/home/models/Qwen3.6-35B-A3B-NVFP4/f$i"; done

cat > "$T/home/freetoken-venv/bin/ft" <<'EOF'
#!/usr/bin/env bash
if [ "${1:-}" = "--version" ]; then echo "freetoken version 0.1.3+g0d652e73a"; exit 0; fi
echo "FAKE_FT_OK"
for a in "$@"; do echo "ARG|$a"; done
EOF

cat > "$T/home/freetoken-venv/bin/python" <<'EOF'
#!/usr/bin/env bash
case "${2:-}" in
  *flashinfer*)
      if [ "$(cat "$STUB_FI_MODE" 2>/dev/null)" = "fail" ]; then
          echo "RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist" >&2
          exit 1
      fi
      echo "CUDA_HOME = /usr/local/cuda"; exit 0 ;;
  *)  echo "[通过] Triton 驱动模块编译 / 加载成功"; exit 0 ;;
esac
EOF
chmod +x "$T/home/freetoken-venv/bin/ft" "$T/home/freetoken-venv/bin/python"
printf '#!/usr/bin/env bash\nexit 1\n' > "$T/bin/sudo"; chmod +x "$T/bin/sudo"
printf '#!/usr/bin/env bash\n' > "$T/bin/gcc"; chmod +x "$T/bin/gcc"

# 假的已装 freetoken 源码：args.py 带 --attention-backend 与 --cors-origins，注册表带 triton/fi
printf 'parser.add_argument(\n    "--attention-backend",\n    type=str,\n)\nparser.add_argument(\n    "--cors-origins",\n    type=str,\n)\n' > "$SP/freetoken/server/args.py"
printf 'SUPPORTED_ATTENTION_BACKENDS.register(\n    "trition_placeholder",\n)\nSUPPORTED_ATTENTION_BACKENDS.register(\n    "triton",\n)\nSUPPORTED_ATTENTION_BACKENDS.register(\n    "fi",\n)\n' > "$SP/freetoken/attention/__init__.py"

export STUB_FI_MODE="$T/fi_mode"; echo ok > "$STUB_FI_MODE"
FAIL=0

run() {   # run <期望退出码> <场景名> <参数...>
    local want="$1" name="$2"; shift 2
    local out rc
    out="$(PATH="$T/bin:/usr/bin:/bin" HOME="$T/home" bash "$SCRIPT" "$@" 2>&1)"; rc=$?
    if [ "$rc" = "$want" ]; then echo "  [通过] $name（退出码 $rc）"; else echo "  [失败] $name：期望 $want 实得 $rc"; FAIL=$((FAIL+1)); fi
    printf '%s\n' "$out" > "$T/last.log"
}
# 注意用 -e 传模式：模式本身以 " -" 开头时，直接 "$2" 会被 grep 当成选项；
# 若写成 grep -q -- "$2" 则文件参数会被吞掉、grep 转而读 stdin → 永久挂住。
check() { if grep -qF -e "$2" "$T/last.log"; then echo "  [通过] $1"; else echo "  [失败] $1"; FAIL=$((FAIL+1)); fi; }
nocheck() { if grep -qF -e "$2" "$T/last.log"; then echo "  [失败] $1"; FAIL=$((FAIL+1)); else echo "  [通过] $1"; fi; }
# 只看「实际要执行的命令行」那一行 —— 提示文案里也会出现同样的参数名，
# 直接 grep 整个日志会误判。
cmd_has_flag() { grep -F -e 'serve --model' "$T/last.log" | tail -1 | grep -qF -e "$1"; }

echo "########## A. 默认后端 = triton ##########"
run 0 "默认不阻断" --dry-run
cmd_has_flag "--attention-backend triton" && echo "  [通过] 命令行带 triton" || { echo "  [失败] 命令行带 triton"; FAIL=$((FAIL+1)); }
check "说明 triton 免 nvcc" "不需要 nvcc"

echo; echo "########## B. 显式 fi + 探针失败 ##########"
echo fail > "$STUB_FI_MODE"
run 1 "fi 缺工具链 -> 阻断" --dry-run --attention-backend fi
check "报出根因" "找不到 CUDA 工具链"
check "给出零下载出路 a)" "attention-backend triton"
check "给出装 toolkit 出路 b)" "cuda-toolkit-13-0"

echo; echo "########## C. 显式 fi + 探针成功 ##########"
echo ok > "$STUB_FI_MODE"
run 0 "fi 工具链就绪 -> 放行" --dry-run --attention-backend fi
cmd_has_flag "--attention-backend fi" && echo "  [通过] 命令行尊重用户选择" || { echo "  [失败] 命令行尊重用户选择"; FAIL=$((FAIL+1)); }

echo; echo "########## D. 等号形式 =auto ##########"
echo fail > "$STUB_FI_MODE"
run 1 "等号形式解析正确且 auto 也查 flashinfer" --dry-run --attention-backend=auto
echo ok > "$STUB_FI_MODE"

echo; echo "########## E. 非法后端名 ##########"
run 2 "非法值 exit 2" --attention-backend bogus
check "给了可选值提示" "不支持的 --attention-backend"

echo; echo "########## F. 缺参数 ##########"
run 2 "缺参数 exit 2" --attention-backend

echo; echo "########## G. 环境变量 FT_ATTN_BACKEND ##########"
out="$(PATH="$T/bin:/usr/bin:/bin" HOME="$T/home" FT_ATTN_BACKEND=fi bash "$SCRIPT" --dry-run 2>&1)"
printf '%s\n' "$out" > "$T/last.log"
grep -q -- "--attention-backend fi" "$T/last.log" && echo "  [通过] 环境变量生效" || { echo "  [失败] 环境变量未生效"; FAIL=$((FAIL+1)); }

echo; echo "########## H. --skip-preflight 仍查后端依赖 ##########"
echo fail > "$STUB_FI_MODE"
run 1 "跳过 Triton 预检也不放过 flashinfer 缺失" --dry-run --skip-preflight --attention-backend fi
echo ok > "$STUB_FI_MODE"

echo; echo "########## I. 参数透传 ##########"
out="$(PATH="$T/bin:/usr/bin:/bin" HOME="$T/home" bash "$SCRIPT" 2>&1)"
printf '%s\n' "$out" > "$T/last.log"
grep -q "FAKE_FT_OK" "$T/last.log" && echo "  [通过] 确实调用了 ft" || { echo "  [失败] 没调用 ft"; FAIL=$((FAIL+1)); }
grep -q "ARG|--attention-backend" "$T/last.log" && echo "  [通过] 参数已透传" || { echo "  [失败] 参数未透传"; FAIL=$((FAIL+1)); }

echo; echo "########## J. 本机版 freetoken 不认该参数 ##########"
cp "$SP/freetoken/server/args.py" "$T/args.bak"
printf 'parser.add_argument("--model")\n' > "$SP/freetoken/server/args.py"
run 0 "降级为不带该参数启动（不阻断）" --dry-run
check "警告参数不存在" "没有 --attention-backend 参数"
cmd_has_flag "--attention-backend" && { echo "  [失败] 命令行仍带该参数"; FAIL=$((FAIL+1)); } || echo "  [通过] 命令行确实不带该参数"
mv "$T/args.bak" "$SP/freetoken/server/args.py"

echo; echo "########## K. 注册表里没有该后端 ##########"
cp "$SP/freetoken/attention/__init__.py" "$T/attn.bak"
printf 'SUPPORTED_ATTENTION_BACKENDS.register("fi",)\n' > "$SP/freetoken/attention/__init__.py"
run 0 "降级为不带该参数启动（不阻断）" --dry-run
check "警告注册表里没有 triton" "后端注册表里没有"
mv "$T/attn.bak" "$SP/freetoken/attention/__init__.py"

echo; echo "########## L. CORS 默认放开 ##########"
run 0 "默认不阻断" --dry-run
cmd_has_flag "--cors-origins *" && echo "  [通过] 命令行带 --cors-origins '*'" || { echo "  [失败] 命令行未带 --cors-origins '*'"; FAIL=$((FAIL+1)); }
check "提示 CORS 已放开" "CORS 已放开"

echo; echo "########## M. CORS 自定义白名单 ##########"
run 0 "自定义 origin 放行" --dry-run --cors-origins 'http://127.0.0.1:8080'
cmd_has_flag "--cors-origins http://127.0.0.1:8080" && echo "  [通过] 白名单已透传" || { echo "  [失败] 白名单未透传"; FAIL=$((FAIL+1)); }
check "提示白名单内容" "CORS 白名单"

echo; echo "########## N. CORS 显式关闭（空串）##########"
run 0 "空串=关掉 CORS 中间件" --dry-run --cors-origins ''
check "提示 CORS 已关闭" "CORS 已关闭"

echo; echo "########## O. CORS 缺参数 ##########"
run 2 "缺参数 exit 2" --cors-origins

echo; echo "########## P. 环境变量 FT_CORS_ORIGINS ##########"
out="$(PATH="$T/bin:/usr/bin:/bin" HOME="$T/home" FT_CORS_ORIGINS='http://localhost:3000' bash "$SCRIPT" --dry-run 2>&1)"
printf '%s\n' "$out" > "$T/last.log"
cmd_has_flag "--cors-origins http://localhost:3000" && echo "  [通过] 环境变量生效" || { echo "  [失败] 环境变量未生效"; FAIL=$((FAIL+1)); }

echo; echo "########## Q. 本机版 freetoken 不认 --cors-origins ##########"
cp "$SP/freetoken/server/args.py" "$T/args2.bak"
printf 'parser.add_argument(\n    "--attention-backend",\n    type=str,\n)\n' > "$SP/freetoken/server/args.py"
run 0 "降级为不带该参数启动（不阻断）" --dry-run
check "警告参数不存在" "没有 --cors-origins 参数"
cmd_has_flag "--cors-origins" && { echo "  [失败] 命令行仍带该参数"; FAIL=$((FAIL+1)); } || echo "  [通过] 命令行确实不带该参数"
cp "$T/args2.bak" "$SP/freetoken/server/args.py"

echo; echo "########## R. KV 自动调优（默认开启）##########"
run 0 "默认不阻断" --dry-run
check "提示自动调优" "自动调优"
check "点明出厂切分只有 8268" "8268 token"

echo; echo "########## S. --kv-context 0 关闭 ##########"
run 0 "关闭后仍可启动" --dry-run --kv-context 0
check "提示已关闭" "已关闭"

echo; echo "########## T. --kv-context 非数字 ##########"
run 2 "非数字 exit 2" --dry-run --kv-context abc
check "报错点名字段" "KV_CONTEXT"
nocheck "非数字时不带病启动" "自动调优"

echo; echo "########## U. --kv-context 缺参数 ##########"
run 2 "缺参数 exit 2" --dry-run --kv-context

echo; echo "########## V. 环境变量 FT_KV_CONTEXT ##########"
out="$(PATH="$T/bin:/usr/bin:/bin" HOME="$T/home" FT_KV_CONTEXT='12288' bash "$SCRIPT" --dry-run 2>&1)"
printf '%s\n' "$out" > "$T/last.log"
check "环境变量生效" "调到 12288"

echo; echo "===== 失败项合计：${FAIL} ====="
