#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ft_gateway.py 自测 —— 两部分：
  A. 注入逻辑单测（纯函数，不起网络）
  B. 端到端代理测试（起一个假上游，真的走一遍 HTTP + SSE 流式透传）

零依赖，直接跑：
    python tests/selftest_gateway.py
"""

from __future__ import annotations

import io
import json
import os
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import ft_gateway as gw  # noqa: E402

通过 = 0
失败 = 0
记录: list[str] = []


def 断言(名称: str, 条件: bool, 详情: str = "") -> None:
    global 通过, 失败
    if 条件:
        通过 += 1
        记录.append(f"  [通过] {名称}")
    else:
        失败 += 1
        记录.append(f"  [失败] {名称}" + (f"  → {详情}" if 详情 else ""))


def 段(标题: str) -> None:
    记录.append("")
    记录.append(f"──── {标题} ────")


# ============================================================ A. 注入逻辑单测

def 测注入() -> None:
    段("A. 注入逻辑单测")

    # A1 普通请求 → 注入 thinking
    体 = json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}]}).encode()
    新, 标 = gw.注入关思考("/v1/chat/completions", 体)
    对 = json.loads(新)
    断言("A1 普通请求被注入 thinking", 对.get("thinking") == {"type": "disabled"}, str(对.get("thinking")))
    断言("A1 注入标签正确", 标 == 'thinking={"type":"disabled"}', 标)

    # A2 原有字段不丢
    断言("A2 model 字段保留", 对.get("model") == "m")
    断言("A2 messages 字段保留", 对.get("messages") == [{"role": "user", "content": "hi"}])

    # A3 客户端已用 chat_template_kwargs 表态 → 不插嘴
    体 = json.dumps({
        "model": "m",
        "messages": [],
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    新, 标 = gw.注入关思考("/v1/chat/completions", 体)
    对 = json.loads(新)
    断言("A3 模板参数已表态则不注入", 标.startswith("跳过") and "thinking" not in 对, 标)

    # A4 客户端已用 thinking 表态（且是 enabled）→ 不插嘴
    体 = json.dumps({"model": "m", "messages": [], "thinking": {"type": "enabled"}}).encode()
    _, 标 = gw.注入关思考("/v1/chat/completions", 体)
    断言("A4 thinking=enabled 被尊重", 标.startswith("跳过"), 标)

    # A5 已带 reasoning_effort → 不插嘴
    体 = json.dumps({"model": "m", "messages": [], "reasoning_effort": "high"}).encode()
    _, 标 = gw.注入关思考("/v1/chat/completions", 体)
    断言("A5 reasoning_effort 已表态则跳过", 标.startswith("跳过"), 标)

    # A6 /v1/messages（Anthropic）同形注入
    体 = json.dumps({"model": "m", "messages": []}).encode()
    新, _ = gw.注入关思考("/v1/messages", 体)
    断言("A6 Anthropic 路径注入 thinking", json.loads(新).get("thinking") == {"type": "disabled"})

    # A7 /v1/responses 用 reasoning.effort
    体 = json.dumps({"model": "m", "input": "hi"}).encode()
    新, 标 = gw.注入关思考("/v1/responses", 体)
    对 = json.loads(新)
    断言("A7 Responses 路径注入 reasoning", 对.get("reasoning") == {"effort": "none"}, str(对.get("reasoning")))
    断言("A7 Responses 不注入 thinking", "thinking" not in 对)

    # A8 非 JSON / 非对象 → 原样放过，不炸
    原 = b"not json at all"
    新, 标 = gw.注入关思考("/v1/chat/completions", 原)
    断言("A8 非 JSON 原样返回", 新 == 原 and 标.startswith("跳过"), 标)

    新, 标 = gw.注入关思考("/v1/chat/completions", b"[1,2,3]")
    断言("A8 非对象原样返回", 标.startswith("跳过"), 标)

    # A9 中文不被转义成 \uXXXX（保持可读、字节更省）
    体 = json.dumps({"model": "m", "messages": [{"role": "user", "content": "中文测试"}]}).encode()
    新, _ = gw.注入关思考("/v1/chat/completions", 体)
    断言("A9 中文保持原样编码", "中文测试".encode("utf-8") in 新)


# ============================================================ B. 端到端代理

收到请求: list[dict] = []


class 假上游(BaseHTTPRequestHandler):
    """模拟 ft serve：回显收到的请求体，并支持 SSE 流式。"""

    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/v1/models":
            数据 = json.dumps({"object": "list", "data": [{"id": "Qwen3.6-35B-A3B-NVFP4"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(数据)))
            self.end_headers()
            self.wfile.write(数据)
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def do_POST(self):
        长度 = int(self.headers.get("Content-Length") or 0)
        体 = json.loads(self.rfile.read(长度)) if 长度 else {}
        收到请求.append(体)

        if 体.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            for 词 in ("你", "好", "世界"):
                帧 = {"choices": [{"delta": {"content": 词}}]}
                self.wfile.write(f"data: {json.dumps(帧, ensure_ascii=False)}\n\n".encode())
                self.wfile.flush()
                time.sleep(0.02)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            数据 = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(数据)))
            self.end_headers()
            self.wfile.write(数据)


def 起服务(处理器, 端口: int) -> ThreadingHTTPServer:
    服务 = ThreadingHTTPServer(("127.0.0.1", 端口), 处理器)
    服务.daemon_threads = True
    threading.Thread(target=服务.serve_forever, daemon=True).start()
    return 服务


def 测代理() -> None:
    段("B. 端到端代理（真 HTTP + SSE）")

    上游端口 = 18777
    网关端口 = 18778
    上游 = 起服务(假上游, 上游端口)

    gw.网关处理器.上游主机 = "127.0.0.1"
    gw.网关处理器.上游端口 = 上游端口
    gw.网关处理器.是否注入 = True
    gw.网关处理器.页面路径 = Path(__file__).resolve().parent.parent / "tools" / "chat.html"
    网关 = 起服务(gw.网关处理器, 网关端口)
    time.sleep(0.3)

    基址 = f"http://127.0.0.1:{网关端口}"

    try:
        # B1 首页返回 chat.html
        with urllib.request.urlopen(基址 + "/", timeout=10) as r:
            正文 = r.read().decode("utf-8")
        断言("B1 首页返回 200", True)
        断言("B1 内容是聊天页面", "FreeToken" in 正文 and "<html" in 正文.lower(), 正文[:60])
        断言("B1 Content-Type 正确",
             "text/html" in r.headers.get("Content-Type", ""), r.headers.get("Content-Type", ""))

        # B2 非流式请求 → 上游应收到注入后的 thinking
        收到请求.clear()
        请求 = urllib.request.Request(
            基址 + "/v1/chat/completions",
            data=json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}]}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(请求, timeout=10) as r:
            r.read()
        断言("B2 非流式转发成功", len(收到请求) == 1, str(len(收到请求)))
        if 收到请求:
            断言("B2 上游收到注入的 thinking",
                 收到请求[0].get("thinking") == {"type": "disabled"}, str(收到请求[0].get("thinking")))
            断言("B2 原字段完整",
                 收到请求[0].get("model") == "m" and len(收到请求[0].get("messages", [])) == 1)

        # B3 SSE 流式透传
        收到请求.clear()
        请求 = urllib.request.Request(
            基址 + "/v1/chat/completions",
            data=json.dumps({"model": "m", "messages": [], "stream": True}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(请求, timeout=10) as r:
            流 = r.read().decode("utf-8")
        断言("B3 流式转发成功", len(收到请求) == 1)
        if 收到请求:
            断言("B3 流式请求也被注入", 收到请求[0].get("thinking") == {"type": "disabled"})
        断言("B3 SSE 三帧都到齐", 流.count("data:") == 4, f"data: 出现 {流.count('data:')} 次")
        断言("B3 收到 [DONE]", "[DONE]" in 流)
        # 注意：SSE 是逐帧发的，三帧分别带「你」「好」「世界」——
        # 所以不能断言 "你好" 连续出现（它从来不会连在一起）。
        断言("B3 中文内容未损坏",
             all(词 in 流 for 词 in ("你", "好", "世界")), 流[:120])

        # B4 GET /v1/models 透传
        with urllib.request.urlopen(基址 + "/v1/models", timeout=10) as r:
            模型 = json.loads(r.read().decode())
        断言("B4 /v1/models 透传正常", 模型["data"][0]["id"] == "Qwen3.6-35B-A3B-NVFP4")

        # B5 客户端已表态 → 网关不覆盖
        收到请求.clear()
        请求 = urllib.request.Request(
            基址 + "/v1/chat/completions",
            data=json.dumps({
                "model": "m", "messages": [],
                "thinking": {"type": "enabled"},
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(请求, timeout=10) as r:
            r.read()
        if 收到请求:
            断言("B5 客户端的 thinking=enabled 未被改写",
                 收到请求[0].get("thinking") == {"type": "enabled"},
                 str(收到请求[0].get("thinking")))

        # B6 上游不可达 → 清晰的 502 而非堆栈
        网关3 = 起服务(gw.网关处理器, 18779)
        gw.网关处理器.上游端口 = 18999          # 无人监听
        time.sleep(0.2)
        请求 = urllib.request.Request(
            "http://127.0.0.1:18779/v1/chat/completions",
            data=b'{"model":"m"}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(请求, timeout=10)
            断言("B6 上游不可达返回 502", False, "居然成功了")
        except urllib.error.HTTPError as e:
            断言("B6 上游不可达返回 502", e.code == 502, str(e.code))
            体 = e.read().decode("utf-8")
            断言("B6 给出可执行的排查提示", "ft ctl stats" in 体, 体[:80])
        网关3.shutdown()

    finally:
        网关.shutdown()
        上游.shutdown()


# ============================================================ C. 端口占用诊断

def 测端口冲突() -> None:
    段("C. 端口占用诊断")

    占用端口 = 18780
    占位 = 起服务(假上游, 占用端口)

    try:
        # C1 端口被占用 → 返回错误，而不是把 traceback 甩给用户
        服务, 错 = gw.创建服务("127.0.0.1", 占用端口)
        断言("C1 端口被占用时返回错误而非抛异常",
             服务 is None and isinstance(错, OSError), f"{服务!r} / {错!r}")
        if 服务 is not None:
            服务.server_close()

        # C2 空闲端口能正常绑定
        服务2, 错2 = gw.创建服务("127.0.0.1", 18781)
        断言("C2 空闲端口绑定成功", 服务2 is not None and 错2 is None, repr(错2))
        if 服务2 is not None:
            服务2.server_close()

        # C3 占用者定位：Linux 上应点名到进程，其他平台安全返回空表
        命中 = gw.谁占用端口(占用端口)
        if sys.platform.startswith("linux"):
            断言("C3 Linux 上定位到占用进程", len(命中) > 0, str(命中))
            断言("C3 占用者就是本测试进程",
                 any(进程号 == os.getpid() for 进程号, _ in 命中), str(命中))
        else:
            断言("C3 非 Linux 平台安全返回空表（不报错）", 命中 == [], str(命中))

        # C4 诊断文本可读、可执行
        缓冲 = io.StringIO()
        原出 = sys.stdout
        sys.stdout = 缓冲
        try:
            gw.诊断端口冲突("127.0.0.1", 占用端口, OSError(98, "Address already in use"))
        finally:
            sys.stdout = 原出
        文本 = 缓冲.getvalue()
        断言("C4 诊断不抛异常且点明端口", f":{占用端口}" in 文本, 文本[:60])
        断言("C4 诊断给出换端口方案", "--port" in 文本)
        断言("C4 诊断给出排查命令", "ss -lntp" in 文本)
        if sys.platform.startswith("linux"):
            断言("C4 诊断点名占用进程 PID", "PID" in 文本)

        # C5 上游探测
        断言("C5 有服务监听时判为在线", gw.上游在监听("127.0.0.1", 占用端口) is True)
        断言("C5 无服务时判为离线", gw.上游在监听("127.0.0.1", 18998, 超时=0.3) is False)

        # C6 默认端口必须避开 ft serve 的主端口及其相邻内部端口
        # （实测 1919 启动后 1920 也被它占着 —— 网关曾经默认就是 1920，撞了）
        默认 = gw.建解析器().parse_args([])
        间距 = abs(默认.port - 默认.upstream)
        断言("C6 默认端口不等于上游端口", 默认.port != 默认.upstream, str(默认.port))
        断言("C6 默认端口避开 1919 / 1920", 默认.port not in (1919, 1920), str(默认.port))
        断言("C6 默认端口与上游拉开 >20 的距离", 间距 > 20, f"{默认.port} vs {默认.upstream}")
    finally:
        占位.shutdown()


# ============================================================ 主流程

def main() -> int:
    print()
    print("═" * 68)
    print(" ft_gateway.py 自测")
    print("═" * 68)

    测注入()
    测代理()
    测端口冲突()

    print("\n".join(记录))
    print()
    print("═" * 68)
    print(f" 通过 {通过} / {通过 + 失败}" + (f"　失败 {失败}" if 失败 else "　全部通过"))
    print("═" * 68)
    return 1 if 失败 else 0


if __name__ == "__main__":
    sys.exit(main())
