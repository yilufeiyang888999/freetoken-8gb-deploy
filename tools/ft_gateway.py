#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FreeToken 本地网关 —— 一次解决「IDE 接入」与「网页聊天」两件事

它解决的问题
------------
FreeToken 的 API 服务（默认 127.0.0.1:1919）本身够用，但直接给浏览器和 IDE 用
会撞上两堵墙：

  ① CORS —— 服务的跨源白名单只放行 Tauri 桌面端的三个 origin
     （`args.py:68`：tauri://localhost,http://tauri.localhost,http://localhost:1420）。
     从 file:// 打开的网页（Origin: null）或别的端口都会被浏览器拦掉。

  ② 关思考传不进去 —— Qwen3.6 默认先输出思维链。而「关思考」这个开关只能放在
     请求体里（`chat_template_kwargs` / `thinking` / `reasoning_effort`），
     偏偏 Cline、Roo Code 这些主流 IDE 扩展只给 Base URL + API Key + Model ID
     三个输入框，没有自定义请求体的入口。于是每个请求都要先烧几百 token 思考 ——
     在 16384 的上下文里，这直接把可用空间吃掉一大块。

本网关把这两个问题一起解决：它**同时**是对外的 API 端点和对内的网页主机，
所以浏览器侧同源（无 CORS 问题），IDE 侧由它统一注入关思考字段。

    浏览器  http://127.0.0.1:1950/           ┐
                                              ├─→ 网关（同源 / 自动注入）─→ ft serve :1919
    IDE     http://127.0.0.1:1950/v1         ┘

用法
----
    # WSL 里跑（服务需已启动）
    python3 tools/ft_gateway.py

    # 常用参数
    --port 1950              网关监听端口（默认 1950）
    --upstream 1919          上游 ft serve 端口（默认 1919）
    --host 127.0.0.1         监听地址（默认只监听本机）
    --keep-thinking          不注入关思考，原样转发（想看思考过程时用）
    --html <路径>            自定义聊天页面（默认取同目录的 chat.html）

端口被占用
----------
启动时报这个错：

    OSError: [Errno 98] Address already in use

含义很直白：**已经有别的进程占着这个端口在监听**（TIME_WAIT 不算，
socket 已开 SO_REUSEADDR）。两种占用者要分清：

| 占用者 | 说明 |
|---|---|
| **ft serve 自己** | **实测：服务启动后，主端口 `1919` 旁边的 `1920` 也被它占着**（supervisor 与 backend worker 之间的内部通道）。kill 掉 1920 上的进程，ft serve 会因为 worker 丢失而整体退出 —— 所以网关**不要用 1920**，默认已挪到 `1950` |
| 上一次的网关没退干净 | 另一个终端窗口里还跑着，或当初用了 `&` / `nohup`；按了 `Ctrl+Z`（挂起）而不是 `Ctrl+C`（退出）也归这类 |

确认是谁占的：

    ss -lntp | grep -E ':(1919|1920)'

脚本在绑定前会扫 `/proc` 找出占用者，直接把 PID 和命令行打出来，
不用自己 `ss` / `lsof` 折腾。

换端口不影响聊天页面：`chat.html` 走的是同源相对路径
（`location.origin + "/v1"`），自动跟随网关端口。只有 IDE 里的
Base URL 需要跟着改。

注入方式
--------
优先用**协议标准字段**而不是模板私有参数，因为它在三套协议里语义一致、字段最短：

    OpenAI   /v1/chat/completions   →  thinking: {"type": "disabled"}
    Anthropic /v1/messages          →  thinking: {"type": "disabled"}
    Responses /v1/responses         →  reasoning: {"effort": "none"}

（依据：`server/openai_api.py:48-70`、`server/anthropic_api.py:297-301`、
 `server/responses_api.py:237-239`、`server/model_meta.py:89-90`。）

**尊重客户端的显式意图**：如果请求里已经带了任何一个思考相关字段
（enable_thinking / thinking / thinking_mode / reasoning_effort），网关不再插手。

零第三方依赖：只用标准库。
"""

from __future__ import annotations

import argparse
import errno
import http.client
import json
import os
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# ---------------------------------------------------------------- 常量

# 这三条路径的请求体会被检查是否注入关思考
注入路径 = ("/v1/chat/completions", "/v1/messages", "/v1/responses")

# 与 freetoken 的 _THINKING_KWARG_KEYS 对齐（model_meta.py:89）：
# 只要出现其中任何一个，就说明客户端已自行表态，网关不再插嘴。
思考相关键 = ("enable_thinking", "thinking", "thinking_mode", "reasoning_effort")

# 逐跳首部，按 RFC 7230 不应转发
逐跳首部 = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade",
})


# ---------------------------------------------------------------- 注入逻辑


def 注入关思考(路径: str, 请求体: bytes) -> tuple[bytes, str]:
    """往请求体里塞一个协议级的关思考字段。

    返回 (新请求体, 说明)。说明用于日志；未改动时以 "跳过" 开头。
    """
    try:
        对象 = json.loads(请求体)
    except Exception:
        return 请求体, "跳过:非 JSON"
    if not isinstance(对象, dict):
        return 请求体, "跳过:非对象"

    # ① 客户端已经明确表态 → 尊重它，网关不插嘴
    模板参数 = 对象.get("chat_template_kwargs")
    if isinstance(模板参数, dict) and any(k in 模板参数 for k in 思考相关键):
        return 请求体, "跳过:模板参数已表态"

    思考 = 对象.get("thinking")
    if isinstance(思考, dict) and 思考.get("type") in ("enabled", "disabled"):
        return 请求体, "跳过:thinking 已表态"

    if isinstance(对象.get("reasoning_effort"), str) and 对象["reasoning_effort"]:
        return 请求体, "跳过:reasoning_effort 已表态"

    # ② 按协议选字段：优先用协议标准字段，而不是某个模板的私有参数
    if 路径 == "/v1/responses":
        对象["reasoning"] = {"effort": "none"}
        标签 = 'reasoning={"effort":"none"}'
    else:
        # /v1/chat/completions 与 /v1/messages 同形（DeepSeek 线格式）
        对象["thinking"] = {"type": "disabled"}
        标签 = 'thinking={"type":"disabled"}'

    return json.dumps(对象, ensure_ascii=False).encode("utf-8"), 标签


# ---------------------------------------------------------------- 端口诊断


def 谁占用端口(端口: int) -> list[tuple[int, str]]:
    """在 Linux 上扫 /proc 找出正在监听指定端口的进程。

    思路：/proc/net/tcp{,6} 里 LISTEN 状态的 socket 带 inode，再去
    /proc/<pid>/fd/* 里找软链接指向 `socket:[inode]` 的进程。

    返回 [(pid, 命令行), ...]。非 Linux、或权限不足读不到别人进程的
    fd 时返回空表 —— 调用方据此退回到"给命令让用户自己查"。
    """
    if not sys.platform.startswith("linux"):
        return []

    目标端口 = f"{端口:04X}"
    索引: set[str] = set()

    for 表 in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            文件 = open(表, "r", encoding="utf-8", errors="replace")
        except OSError:
            continue
        with 文件:
            next(文件, None)                    # 跳过表头
            for 行 in 文件:
                列 = 行.split()
                # 列: 0序号 1本地 2远端 3状态 4队列 5定时 6重传 7uid 8超时 9inode
                if len(列) < 10 or 列[3] != "0A":       # 0A = TCP_LISTEN
                    continue
                if 列[1].rsplit(":", 1)[-1].upper() != 目标端口:
                    continue
                索引.add(列[9])

    if not 索引:
        return []

    命中: list[tuple[int, str]] = []
    try:
        进程列表 = list(Path("/proc").iterdir())
    except OSError:
        return []

    for 进程目录 in 进程列表:
        if not 进程目录.name.isdigit():
            continue
        try:
            描述符 = list((进程目录 / "fd").iterdir())
        except OSError:                          # 别人的进程 / 已退出
            continue
        for 链接 in 描述符:
            try:
                指向 = os.readlink(链接)
            except OSError:
                continue
            if 指向.startswith("socket:[") and 指向[8:-1] in 索引:
                try:
                    原始 = (进程目录 / "cmdline").read_bytes()
                except OSError:
                    原始 = b""
                文本 = 原始.replace(b"\0", b" ").decode("utf-8", "replace").strip()
                命中.append((int(进程目录.name), 文本 or "(读不到命令行)"))
                break
    return 命中


def 上游在监听(主机: str, 端口: int, 超时: float = 1.5) -> bool:
    """只探 TCP 能不能连上 —— 不管上游是否还在加载权重。"""
    try:
        with socket.create_connection((主机, 端口), timeout=超时):
            return True
    except OSError:
        return False


def 端口已被占用(主机: str, 端口: int) -> bool:
    """用不带 SO_REUSEADDR 的裸 socket 探测端口是否已被监听。

    不能拿 ThreadingHTTPServer 的绑定结果当判据：HTTPServer 会设
    SO_REUSEADDR，而 **Windows 的 SO_REUSEADDR 语义是"允许两个 socket
    绑同一个 addr:port"**（Linux 只对 TIME_WAIT 放行）。于是在 Windows 上
    端口冲突会悄无声息地过去 —— 两个进程都"启动成功"，谁先绑的谁收流量，
    另一个永远收不到请求，还不报错。裸 socket 默认不开这个选项，判据
    在两个平台上才一致。
    """
    try:
        地址族 = socket.getaddrinfo(主机, 端口, type=socket.SOCK_STREAM)[0][0]
    except OSError:
        地址族 = socket.AF_INET

    探测 = socket.socket(地址族, socket.SOCK_STREAM)
    try:
        探测.bind((主机, 端口))
        return False
    except OSError:
        return True
    finally:
        探测.close()


def 创建服务(主机: str, 端口: int):
    """尝试绑定端口。成功返回 (服务, None)，失败返回 (None, 异常)。"""
    if 端口已被占用(主机, 端口):
        return None, OSError(errno.EADDRINUSE, "Address already in use")
    try:
        return ThreadingHTTPServer((主机, 端口), 网关处理器), None
    except OSError as 错:
        return None, 错


def 诊断端口冲突(主机: str, 端口: int, 错: OSError) -> None:
    """把 Address already in use 翻译成"谁占的、怎么处置"。"""
    脚本名 = Path(__file__).name
    备用端口 = 端口 + 1

    print()
    print("═" * 68)
    print(f" 启动失败：{主机}:{端口} 已被占用")
    print("═" * 68)
    print(f"  系统错误    {错}")
    print()

    占用者 = 谁占用端口(端口)
    if 占用者:
        print("  占用这个端口的进程：")
        for 进程号, 命令行 in 占用者:
            短 = 命令行 if len(命令行) <= 94 else 命令行[:91] + "..."
            print(f"    PID {进程号:<8} {短}")
        print()
        print("  八成是上一次的网关没退干净（另一个终端窗口，或当初用了 & / nohup）。")
        print("  停掉它再重跑：")
        print(f"    kill {占用者[0][0]}                  # 没反应就 kill -9 {占用者[0][0]}")
    else:
        print("  没定位到占用进程 —— 可能属于其他用户，或不在本网络命名空间里。")
        print("  手动查看：")
        print(f"    ss -lntp | grep ':{端口}'")
        print(f"    lsof -i :{端口}                    # 若装了 lsof")

    print()
    第一条 = (f"    ① 停掉占用进程      kill {占用者[0][0]}"
             if 占用者 else
             "    ① 停掉占用进程      先用上面的命令查出 PID，再 kill <PID>")
    print("  三条出路：")
    print(第一条)
    print(f"    ② 换端口起          python3 {脚本名} --port {备用端口}")
    print("        聊天页面不用改（同源请求，跟着网关端口走）")
    print(f"        IDE 里 Base URL 改成  http://{主机}:{备用端口}/v1")
    print(f"    ③ 排除 WSL 外面的程序")
    print(f"        WSL2 是独立网络命名空间，Windows 侧占 {端口} 一般不影响这里；")
    print(f"        要确认，在 PowerShell 里跑： Get-NetTCPConnection -LocalPort {端口}")
    print("═" * 68)
    print()


# ---------------------------------------------------------------- 处理器


class 网关处理器(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # 由 main() 注入
    上游主机 = "127.0.0.1"
    上游端口 = 1919
    是否注入 = True
    页面路径: Path | None = None
    计数锁 = threading.Lock()
    请求计数 = 0

    # ---- 日志：只留一行，别把终端刷爆 ----
    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("  " + (fmt % args) + "\n")
        sys.stderr.flush()

    # ---- 健康检查用的快捷端点 ----
    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path.split("?")[0] in ("/", "/index.html", "/chat.html"):
            return self._送页面()
        self._转发("GET")

    def do_POST(self) -> None:
        self._转发("POST")

    def do_DELETE(self) -> None:
        self._转发("DELETE")

    def do_PUT(self) -> None:
        self._转发("PUT")

    # ---- 静态页面 ----
    def _送页面(self) -> None:
        页面 = self.页面路径
        if not 页面 or not 页面.is_file():
            self.送文本(
                404,
                "找不到聊天页面。\n\n"
                f"期望路径：{页面}\n\n"
                "把 chat.html 放到与 ft_gateway.py 同一目录，或用 --html <路径> 指定。\n",
            )
            return
        数据 = 页面.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(数据)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(数据)

    def 送文本(self, 状态: int, 正文: str) -> None:
        数据 = 正文.encode("utf-8")
        self.send_response(状态)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(数据)))
        self.end_headers()
        self.wfile.write(数据)

    # ---- 核心：转发 ----
    def _转发(self, 方法: str) -> None:
        原始路径 = self.path
        路径 = 原始路径.split("?")[0]

        # 1) 读请求体
        长度 = int(self.headers.get("Content-Length") or 0)
        请求体 = self.rfile.read(长度) if 长度 > 0 else b""

        # 2) 按需注入
        注记 = "原样"
        if self.是否注入 and 请求体 and 路径 in 注入路径:
            请求体, 结果 = 注入关思考(路径, 请求体)
            注记 = ("未注入（" + 结果[3:] + "）") if 结果.startswith("跳过") else ("已注入 " + 结果)

        with 网关处理器.计数锁:
            网关处理器.请求计数 += 1
            序号 = 网关处理器.请求计数
        self.log_message(f"[{序号}] {方法} {原始路径}  →  {注记}")

        # 3) 转发首部（剔除逐跳首部，Host 改指上游）
        首发 = {
            键: 值
            for 键, 值 in self.headers.items()
            if 键.lower() not in 逐跳首部 and 键.lower() not in ("host", "content-length")
        }
        首发["Host"] = f"{self.上游主机}:{self.上游端口}"
        首发["Content-Length"] = str(len(请求体))

        # 4) 发往上游
        try:
            连接 = http.client.HTTPConnection(self.上游主机, self.上游端口, timeout=600)
            连接.request(方法, 原始路径, body=请求体 or None, headers=首发)
            上游响应 = 连接.getresponse()
        except (ConnectionRefusedError, socket.timeout, OSError) as 错:
            self.log_message(f"[{序号}] 上游不可达：{错}")
            self.送文本(
                502,
                f"连不上上游 ft serve（{self.上游主机}:{self.上游端口}）：{错}\n\n"
                "先确认服务在跑：  ft ctl stats\n"
                "或看日志：        ~/models/freetoken-serve.log\n",
            )
            return

        # 5) 回送响应头。流式（SSE）时长度未知，只能用「读完即关连接」告诉客户端结束。
        流式 = "text/event-stream" in (上游响应.getheader("Content-Type") or "")
        self.send_response(上游响应.status)
        for 键, 值 in 上游响应.getheaders():
            低 = 键.lower()
            if 低 in 逐跳首部 or 低 in ("content-length", "connection"):
                continue
            self.send_header(键, 值)
        if not 流式:
            长度值 = 上游响应.getheader("Content-Length")
            if 长度值:
                self.send_header("Content-Length", 长度值)
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        # 6) 透传正文：SSE 逐行刷，普通响应整块送
        try:
            if 流式:
                while True:
                    行 = 上游响应.readline()          # SSE 以换行分帧 → 天然对齐
                    if not 行:
                        break
                    self.wfile.write(行)
                    self.wfile.flush()
            else:
                while True:
                    块 = 上游响应.read(65536)
                    if not 块:
                        break
                    self.wfile.write(块)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # 客户端提前断开（用户点了停止）—— 正常现象，不必报错
            pass
        finally:
            try:
                连接.close()
            except Exception:
                pass


# ---------------------------------------------------------------- 入口


# 默认端口刻意离上游 1919 拉开距离：ft serve 会在主端口相邻位置另开一个内部
# 通道（实测 1919 + 1 = 1920 归它所有），贴着它就是自找冲突。
默认网关端口 = 1950


def 建解析器() -> argparse.ArgumentParser:
    """单独抽出来，方便测试直接验默认值（不必真的起服务）。"""
    解析器 = argparse.ArgumentParser(
        description="FreeToken 本地网关：托管聊天页面 + 转发 API + 自动关思考",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    解析器.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1）")
    解析器.add_argument("--port", type=int, default=默认网关端口,
                        help=f"网关端口（默认 {默认网关端口}）")
    解析器.add_argument("--upstream", type=int, default=1919, help="上游 ft serve 端口（默认 1919）")
    解析器.add_argument("--upstream-host", default="127.0.0.1", help="上游主机（默认 127.0.0.1）")
    解析器.add_argument("--keep-thinking", action="store_true",
                        help="不注入关思考，原样转发（想看模型思考过程时用）")
    解析器.add_argument("--html", default=None, help="聊天页面路径（默认取同目录 chat.html）")
    return 解析器


def main() -> int:
    参数 = 建解析器().parse_args()

    页面 = Path(参数.html) if 参数.html else Path(__file__).with_name("chat.html")

    网关处理器.上游主机 = 参数.upstream_host
    网关处理器.上游端口 = 参数.upstream
    网关处理器.是否注入 = not 参数.keep_thinking
    网关处理器.页面路径 = 页面

    # 先抢端口再看热闹：绑不上就直接报清楚，不打印误导人的启动横幅
    服务, 绑定错 = 创建服务(参数.host, 参数.port)
    if 服务 is None:
        诊断端口冲突(参数.host, 参数.port, 绑定错)
        return 2

    上游在线 = 上游在监听(参数.upstream_host, 参数.upstream)

    print()
    print("═" * 68)
    print(" FreeToken 本地网关")
    print("═" * 68)
    上游状态 = "已就绪" if 上游在线 else "★ 没在监听 —— 先起 ft serve（bash ~/serve_freetoken.sh）"
    页面状态 = str(页面) if 页面.is_file() else str(页面) + "  ← 未找到"

    print(f"  监听        http://{参数.host}:{参数.port}")
    print(f"  上游        http://{参数.upstream_host}:{参数.upstream}  （{上游状态}）")
    print(f"  聊天页面    {页面状态}")
    print(f"  关思考      {'关闭（原样转发）' if 参数.keep_thinking else '开启（自动注入）'}")
    print("─" * 68)
    if 参数.port != 参数.upstream and abs(参数.port - 参数.upstream) <= 20:
        print(f"  ⚠ 端口 {参数.port} 紧贴上游 {参数.upstream} —— ft serve 会在主端口相邻处")
        print(f"    另开内部通道（实测 {参数.upstream}+1={参数.upstream + 1} 归它所有），")
        print(f"    贴着它就是自找冲突。建议换远一点：  --port {参数.upstream + 20}")
        print("─" * 68)
    if not 上游在线:
        print("  提示：上游没起，网关照样能启动 —— 等 ft serve 起来后**不用重启网关**；")
        print("        在那之前，任何请求都会收到 502 加排查提示。")
        print("─" * 68)
    print("  浏览器聊天：")
    print(f"    http://{参数.host}:{参数.port}/")
    print()
    print("  IDE 扩展（Cline / Roo Code → OpenAI Compatible）：")
    print(f"    Base URL   http://{参数.host}:{参数.port}/v1")
    print("    API Key    随便填（本地服务不校验）")
    print("    Model ID   Qwen3.6-35B-A3B-NVFP4")
    print()
    print("  Claude Code（走 Anthropic 协议）：")
    print(f"    ANTHROPIC_BASE_URL=http://{参数.host}:{参数.port}")
    print()
    print("  Ctrl+C 停止")
    print("═" * 68)
    print()

    服务.daemon_threads = True
    try:
        服务.serve_forever()
    except KeyboardInterrupt:
        print("\n  已停止。")
    finally:
        服务.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
