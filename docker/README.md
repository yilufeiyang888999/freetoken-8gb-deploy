# FreeToken 容器化部署方案

> **这份配置的定位：交付形态，不是当前打通链路的路径。**
> 如果你只是想把 FreeToken 跑起来看效果，请走 WSL2 原生安装（`wsl_setup_freetoken.sh`）。
> 这份配置是给"要把 ChatBI / 智能问答 POC 打包交付给客户"那个阶段准备的。

---

## 一、官方支持状况

FreeToken **目前不提供官方 Docker 支持**。这不是猜测，是查证结果：

| 查证项 | 结果 |
|---|---|
| 仓库根目录 | 无 `Dockerfile`、无 `docker/` 目录、无 `compose.yaml` |
| `docs/install.md` 全文 | **不含 Docker 字样**，只给 PyPI / 源码 / nightly wheels 三种装法 |
| 官方镜像仓库 | 无。issue #379 就是请求发布到 `ghcr.io/flashml-org/freetoken`，**未完成** |

社区已经推了一轮，但全部卡在 open：

| Issue | 标题 | 状态 | 创建 |
|---|---|---|---|
| #11 | Docker Support | open | 2026-08-21 |
| #295 | Add Docker container capability | open（含完整 Dockerfile + compose + 文档） | 2026-08-30 |
| #379 | ci(container): publish official Docker images to GHCR | open | 2026-09-04 |
| #380 | ci(container): 配置镜像构建 | open | — |

`#295` 是一份写得很完整的 PR（作者 recrudesce），有人已经在下面催 `can this please be merged in`。
本目录的 `Dockerfile` 和 `docker-compose.yml` 就是在它的基础上做的，并针对 WSL2 + 消费级显卡加固了三处。

**当前你只能自建镜像。** 本目录的配置文件可以直接用。

---

## 二、三个必须知道的故障点

### 🔴 故障点 1：容器里设内存上限，会被静默 OOM-kill

这是最要命的一条，来自官方仓库自己的 issue **#333**（仍未修复）：

> MoE expert-bank loader 的内存检查**只读 `/proc/meminfo` 的 `MemAvailable`**，
> 不读 cgroup 限制。在 `docker run --memory=N` 的容器里，它读到的是**宿主机的空闲内存**，
> 于是按宿主内存去开并行读专家池（瞬时开销多一整个 shard），
> 结果被 cgroup OOM-kill，**且 FreeToken 不输出任何诊断信息**。

issue 里给了复现：容器限 2GB / 6GB，加载需要 9GB，检查函数两次都返回 `True`（错误地认为放得下）。
修复 PR **#334** 也还没合并。

**规避方式：绝不给容器设 `--memory` / `mem_limit`。**
让内存总量由 WSL2 的 `.wslconfig` 统一管。本目录的 compose 文件已经刻意不写内存限制——
这不是遗漏，是故意的。

### 🔴 故障点 2：WSL2 下 pinned memory 受限

NVIDIA 官方 *CUDA on WSL User Guide* 明确列入已知限制：

> **Pinned system memory**（应用为 GPU 访问而常驻的系统内存）**availability for applications is limited.**
> 某些深度学习工作负载可能超出此限制并无法工作。

而 FreeToken 的核心机制——带宽自适应的 CPU-GPU 协同执行——**恰恰重度依赖页锁定内存做零拷贝传输**。
这是 WSL2 这条路线上最本质的损耗，Docker 又会在此之上再叠一层。

缓解手段：
- `.wslconfig` 里把 `memory` 调大（默认只给宿主 50%，会饿死 pinned 池）——本项目模板已设 28GB
- 容器加 `ulimit memlock=-1` 和 `cap_add: IPC_LOCK`——compose 文件已加
- 启动参数带 `--moe-cpu-layers auto`——官方文档注明该模式专为 Windows/WSL 设计，会自动适配 pinned 上限

### 🟡 故障点 3：权重目录放错位置，会慢到不可用

权重必须落在 WSL2 的 ext4 文件系统里。两种错误写法：

| 写法 | 后果 |
|---|---|
| `-v D:/models:/models` | 走 9p 协议跨系统读，18GB 权重能读到几十分钟 |
| `-v /mnt/d/models:/models` | 同上，`/mnt/*` 也是 9p，不是原生文件系统 |
| ✅ 命名卷 / `~/models` | 落在 vhdx 里，原生 ext4，速度正常 |

### 🟡 附带损耗：WSL2 显存税

WSL2 会通过 WDDM 硬件保留和 DXGKRNL 开销吃掉 **8–15% 显存**，这部分 CUDA 应用看不见也拿不回。
你的 RTX 5060 标称 8GB，WSL2 里实际暴露约 7GB——检测脚本实测空闲 7.0GB，和这个数字吻合。

---

## 三、你这台机器的现状（已检测）

| 项 | 实测 | 判定 |
|---|---|---|
| Docker CLI | **29.8.0** | ✅ 版本足够新，`--gpus all` 有稳定支持 |
| Docker Desktop 位置 | `D:\<数据盘> | ✅ 数据盘在 D 盘，空间够（99GB 可用） |
| `~/.docker/daemon.json` | 只有镜像加速器 + buildkit，**无 nvidia runtime** | ⚠️ 需先验证 GPU 直通 |
| Docker 引擎 | **当前未运行** | ⚠️ 需启动 Docker Desktop |
| WSL 发行版 | **只有 `docker-desktop`**，无 Ubuntu | 🔴 挡路：装 nvidia-container-toolkit 需要 Ubuntu |
| C 盘可用 | 34GB | 🔴 镜像约 12–15GB，别往 C 盘放 |

---

## 四、操作步骤

### 前提：先把 Ubuntu 装好

`docker-desktop` 是 Docker 自用的专用发行版，没有正常用户态环境，装不了 toolkit。先：

```powershell
wsl --install -d Ubuntu --location D:\WSL\Ubuntu
```

### 步骤 1：验证 GPU 直通（先做这个，别急着建镜像）

启动 Docker Desktop，然后在 WSL 里执行：

```bash
docker run --rm --gpus all nvidia/cuda:13.3.0-base-ubuntu24.04 nvidia-smi
```

- **打印出显卡表** → 直通正常，继续。Docker Desktop 4.34+ 能自动代理 GPU，不用手工装 toolkit
- **报 `could not select device driver`** → 需要装 toolkit：
  ```bash
  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  # 然后在 Docker Desktop: Settings → Docker Engine 里确认 nvidia runtime，Apply & Restart
  ```

### 步骤 2：构建镜像

```bash
cd <仓库>/docker
docker build -t freetoken:latest .
```

基础镜像 `nvidia/cuda:13.3.0-devel-ubuntu24.04` 是 devel 版，**压包就有 3.4GB**，
展开构建完镜像约 12–15GB。首次构建建议留 30–60 分钟，取决于网速。

> **标签版本的坑（已修正）**：社区 PR #295 里写的是 `13.3.1-devel-ubuntu26.04`，
> 但 NVIDIA 官方 Docker Hub 上 CUDA 13.3 只有 **13.3.0** 这一个版本，**没有 13.3.1**。
> 照抄会报 `manifest unknown` 直接构建失败。本目录已改成核对过的 `13.3.0-devel-ubuntu24.04`。
>
> 备选标签（均已核对存在）：`13.3.0-devel-ubuntu26.04`（PR 作者用的）、`13.3.0-devel-ubuntu22.04`。
>
> 兼容性说明：镜像内是 CUDA 13.3.0 toolkit，你的驱动是 CUDA 13.4 UMD，
> **驱动版本高于 toolkit 是正常且推荐的组合**，不存在兼容问题。
>
> 加速器提示：你 `daemon.json` 里配的 6 个镜像源只加速 Docker Hub 的 library 命名空间，
> `nvidia/cuda` 不一定命中缓存，`ghcr.io/astral-sh/uv` 则完全不走这些源。
> 国内拉这两个都可能很慢，必要时给 ghcr 单独配代理。

### 步骤 3：启动

```bash
docker compose up -d
docker compose logs -f          # 盯着看，首次要下 18GB 权重
```

### 步骤 4：验证

```bash
curl http://127.0.0.1:1919/v1/models

curl http://127.0.0.1:1919/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.6-35B-A3B","messages":[{"role":"user","content":"你好"}]}'
```

WSL2 自带 localhost 转发，Windows 侧的程序直接访问 `http://127.0.0.1:1919` 就行，不用配端口映射。

---

## 五、什么阶段该用这份配置

| 场景 | 建议 | 理由 |
|---|---|---|
| 第一次跑通、看效果 | ❌ 用 WSL2 原生 | Docker 多两层损耗，还多一个未修复的 bug |
| 调优 tok/s、找最优层数配比 | ❌ 用 WSL2 原生 | 容器化会干扰带宽实测，`ft bench bw` 的数字不可信 |
| **打包给客户做 POC 交付** | ✅ 用这份配置 | 环境可复现、一键起服务，这才是 Docker 的正确用法 |
| **部署到客户的 Linux 服务器** | ✅✅ Docker 主场 | 没有 WSL2 这层损耗，pinned memory 不受限 |

**一句话：Docker 解决的是"环境一致性和交付效率"，不是"能不能跑"。**
FreeToken 的瓶颈在内存带宽和 pinned memory 这种物理层，容器化不会给它加性能，只会加一层损耗。

先把 WSL2 原生那条路跑通、拿到实测数据，再决定容器化值不值。
