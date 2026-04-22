import asyncio
import json
import os
import shutil
import socket
import subprocess
import time
from collections.abc import AsyncIterator

import httpx
import psutil

from nanoserve.baselines.base import Backend
from nanoserve.config import REPO_ROOT, ModelSpec


def _free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _find_llama_server() -> str:
    env = os.environ.get("LLAMA_SERVER_BIN")
    if env and os.path.exists(env):
        return env
    for name in ("llama-server", "server"):
        p = shutil.which(name)
        if p:
            return p
    local = REPO_ROOT / "vendor" / "llama.cpp" / "build" / "bin" / "llama-server"
    if local.exists():
        return str(local)
    raise RuntimeError(
        "llama-server binary not found. build llama.cpp (scripts/download_models.sh) "
        "or set LLAMA_SERVER_BIN to the binary path."
    )


class LlamaCppBackend(Backend):
    """runs llama-server in a subprocess and streams completions over http.
    uses the openai-compatible /v1/chat/completions endpoint with stream=true.
    """

    name = "llama_cpp"

    def __init__(self, model: ModelSpec, n_gpu_layers: int = 999, ctx_size: int = 4096):
        self.model_spec = model
        self.n_gpu_layers = n_gpu_layers
        self.ctx_size = ctx_size
        self._proc: subprocess.Popen | None = None
        self._port: int = 0
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        binary = _find_llama_server()
        self._port = _free_port()
        model_path = str(REPO_ROOT / self.model_spec.path) if not os.path.isabs(
            self.model_spec.path
        ) else self.model_spec.path
        if not os.path.exists(model_path):
            raise RuntimeError(
                f"model file missing: {model_path} (run scripts/download_models.sh)"
            )

        cmd = [
            binary,
            "-m", model_path,
            "-c", str(self.ctx_size),
            "-ngl", str(self.n_gpu_layers),
            "--port", str(self._port),
            "--host", "127.0.0.1",
            "--log-disable",
        ]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        await self._wait_ready()
        self._client = httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{self._port}", timeout=httpx.Timeout(120.0)
        )

    async def _wait_ready(self, timeout_s: float = 60.0) -> None:
        deadline = time.time() + timeout_s
        url = f"http://127.0.0.1:{self._port}/health"
        async with httpx.AsyncClient(timeout=1.0) as c:
            while time.time() < deadline:
                try:
                    r = await c.get(url)
                    if r.status_code == 200:
                        return
                except Exception:
                    pass
                await asyncio.sleep(0.25)
        raise RuntimeError("llama-server failed to become ready")

    async def stop(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // 4)

    def get_mem_mb(self) -> float:
        driver = psutil.Process().memory_info().rss
        if self._proc is None:
            return driver / (1024 * 1024)
        try:
            server = psutil.Process(self._proc.pid)
            total = driver + server.memory_info().rss
            for child in server.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except psutil.NoSuchProcess:
                    continue
            return total / (1024 * 1024)
        except psutil.NoSuchProcess:
            return driver / (1024 * 1024)

    async def generate_stream(self, prompt: str, max_new_tokens: int) -> AsyncIterator[str]:
        if self._client is None:
            raise RuntimeError("backend not started")

        payload = {
            "model": self.model_spec.name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
            "stream": True,
        }

        async with self._client.stream(
            "POST", "/v1/chat/completions", json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                piece = delta.get("content")
                if piece:
                    yield piece
