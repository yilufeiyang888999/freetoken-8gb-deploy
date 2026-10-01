# ============================================================================
# FreeToken 容器镜像
# ----------------------------------------------------------------------------
# 参考来源：社区 PR FlashML-org/FreeToken#295（作者 recrudesce，尚未合并）
#           本文件在其基础上做了三处针对 WSL2 + 消费级显卡的加固，见下方标注
#
# 关键设计说明：
#   1. 必须用 devel 版基础镜像，不能用 runtime 版 —— FreeToken 的 CUDA 内核
#      是首次使用时 JIT 编译的，镜像里必须带 nvcc 和 CUDA 头文件。
#      runtime 版镜像不带 nvcc，启动时会直接失败。
#   2. 必须钉 Python 3.11/3.12 —— Python 3.14 与 PyTorch 扩展的 ABI 不兼容，
#      构建时会出现难以定位的编译错误。宿主的 python 版本与此无关，容器内独立。
#   3. setuptools 必须低于 70.0.0 —— 新版 setuptools 移除了部分旧的构建接口，
#      会导致 freetoken 的 C++ 扩展编译失败。
# ============================================================================

# 【加固点 1】devel 版镜像，带 nvcc（runtime 版不带，会直接启动失败）
#
# ⚠️ 标签版本号已修正：社区 PR #295 里写的是 13.3.1，但 Docker Hub 上
#    NVIDIA 官方只有 13.3.0 这一个 CUDA 13.3 版本，13.3.1 不存在，照抄会
#    manifest unknown 构建失败。以下标签已核对官方 Docker Hub 与 NVIDIA
#    GitLab 镜像清单，确认存在（amd64，约 3.4GB）。
#    备选：13.3.0-devel-ubuntu26.04（PR 作者用的那个，同为官方正式标签）
FROM nvidia/cuda:13.3.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH="/opt/venv/bin:/usr/local/cuda/bin:$PATH" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH" \
    CPATH="/usr/local/cuda/include:$CPATH" \
    LIBRARY_PATH="/usr/local/cuda/lib64:$LIBRARY_PATH"

# 编译工具链：build-essential 提供 gcc/g++，ninja 加速 PyTorch 扩展构建
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ninja-build \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 从官方镜像拷贝 uv 二进制，不用 pip 装（更快、且不受 PEP 668 限制）
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 【加固点 2】显式钉 Python 3.11，规避 3.14 的 ABI 问题
# 【加固点 3】setuptools 钉在 70.0.0 以下，规避扩展编译失败
RUN uv venv /opt/venv --python 3.11 && \
    uv pip install --no-cache "setuptools<70.0.0" wheel torch ninja && \
    uv pip install --no-cache "freetoken[accel]"

# 1919 是 ft serve 默认端口
EXPOSE 1919

# 健康检查：接口活着就说明引擎起来了（模型加载中会返回非 200，属正常）
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD curl -sf http://127.0.0.1:1919/v1/models || exit 1

ENTRYPOINT ["ft", "serve"]
CMD ["--host", "0.0.0.0", "--port", "1919", "--moe-cpu-layers", "auto"]
