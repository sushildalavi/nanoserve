"""MLX-backed serving backend (serial generation).

Uses `mlx_lm.load` to pull TinyLlama via huggingface and convert the
weights into MLX's native format on first run. `mlx.nn.quantize` applies
per-group symmetric int4 quantization in-place; on Metal that uses
native int4 matmul kernels, which is the platform capability the
pytorch-MPS path doesn't have.

quant_mode semantics:
- "fp16"  -> no quantization
- "int4"  -> mlx.nn.quantize(group_size=64, bits=4)
- "int8"  -> mlx.nn.quantize(group_size=64, bits=8)

Only serial (concurrency 1) is supported here — matches the Phase 6A
scope of answering "does native int4 flip the sign of the regression?".
Continuous batching under MLX is a separate multi-day port and is out
of scope.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from nanoserve.baselines.base import Backend
from nanoserve.config import ModelSpec


class MLXBackend(Backend):
    name = "mlx"

    def __init__(self, model: ModelSpec, quant_mode: str = "fp16"):
        self.model_spec = model
        self.quant_mode = quant_mode
        self._model = None
        self._tokenizer = None
        # populated from GenerationResponse on each stream completion
        self._last_peak_memory: int = 0
        self._last_prompt_tps: float = 0.0
        self._last_gen_tps: float = 0.0

    async def start(self) -> None:
        def _load():
            # imports here so the top level stays cheap for users who
            # never touch the mlx path.
            from mlx.nn import quantize as mx_quantize
            from mlx_lm import load

            model, tokenizer = load(self.model_spec.path)
            if self.quant_mode in ("int4", "int8"):
                bits = 4 if self.quant_mode == "int4" else 8
                mx_quantize(model, group_size=64, bits=bits)
            elif self.quant_mode not in ("fp16", "none"):
                raise ValueError(f"mlx backend: unknown quant_mode {self.quant_mode!r}")
            return model, tokenizer

        # model load is cpu-bound + does disk / HF network; run in a
        # worker so the event loop isn't blocked.
        self._model, self._tokenizer = await asyncio.to_thread(_load)

    async def stop(self) -> None:
        self._model = None
        self._tokenizer = None

    def count_tokens(self, text: str) -> int:
        if self._tokenizer is None:
            return 0
        tok = self._tokenizer
        # mlx_lm.TokenizerWrapper forwards __call__ to the underlying HF tokenizer
        return len(tok.encode(text)) if hasattr(tok, "encode") else len(tok(text).input_ids)

    async def generate_stream(
        self, prompt: str, max_new_tokens: int
    ) -> AsyncIterator[str]:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("mlx backend not started")

        model = self._model
        tokenizer = self._tokenizer
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        sentinel = object()

        def _producer():
            # mlx_lm streams GenerationResponse objects synchronously;
            # hop each token back to the event loop via call_soon_threadsafe.
            from mlx_lm import stream_generate

            try:
                last_peak = 0
                last_p_tps = 0.0
                last_g_tps = 0.0
                for resp in stream_generate(
                    model, tokenizer, prompt, max_tokens=max_new_tokens
                ):
                    last_peak = resp.peak_memory
                    last_p_tps = resp.prompt_tps
                    last_g_tps = resp.generation_tps
                    if resp.text:
                        loop.call_soon_threadsafe(queue.put_nowait, resp.text)
                    if resp.finish_reason:
                        break
                self._last_peak_memory = last_peak
                self._last_prompt_tps = last_p_tps
                self._last_gen_tps = last_g_tps
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        # kick the producer off in a background thread and drain the
        # queue as tokens arrive. do NOT await the producer future
        # before yielding, otherwise generation is non-streaming.
        producer_task = asyncio.create_task(asyncio.to_thread(_producer))
        try:
            while True:
                item = await queue.get()
                if item is sentinel:
                    break
                yield item
        finally:
            # propagate any producer exception / ensure it is cleaned up
            if not producer_task.done():
                producer_task.cancel()
            try:
                await producer_task
            except (asyncio.CancelledError, BaseException):
                pass

    def get_stats(self) -> dict:
        # mem_peak is reported in bytes by mlx; convert to mb to match
        # the ablation row convention.
        return {
            "mlx_peak_memory_mb": round(self._last_peak_memory / (1024 * 1024), 1),
            "mlx_prompt_tps": round(self._last_prompt_tps, 2),
            "mlx_gen_tps": round(self._last_gen_tps, 2),
        }
