"""Backend adapter that plugs NanoServeEngine into the existing runner.
lets the phase-1 bench harness drive the phase-2 engine without any
runner-side changes. the flag-flipped ablation rows all run through here.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from nanoserve.baselines.base import Backend
from nanoserve.config import ModelSpec
from nanoserve.engine.engine import NanoServeEngine
from nanoserve.engine.service import SubmitRequest


class NanoServeBackend(Backend):
    name = "nanoserve"

    def __init__(
        self,
        model: ModelSpec,
        batching_mode: str = "serial",
        max_batch_size: int = 1,
        quant_mode: str = "none",
    ):
        self.model_spec = model
        self.batching_mode = batching_mode
        self.max_batch_size = max_batch_size
        self.quant_mode = quant_mode
        self._engine: NanoServeEngine | None = None
        self._final_stats: dict = {}

    async def start(self) -> None:
        self._engine = NanoServeEngine(
            self.model_spec,
            max_batch_size=self.max_batch_size,
            batching_mode=self.batching_mode,
            quant_mode=self.quant_mode,
        )
        await self._engine.start()

    async def stop(self) -> None:
        if self._engine is not None:
            # capture scheduler stats before the engine is torn down
            self._final_stats = self._snapshot_stats()
            await self._engine.stop()
            self._engine = None

    def _snapshot_stats(self) -> dict:
        stats = self._engine._scheduler.stats
        total_forwards = (
            self._engine.batched_forward_steps + self._engine.single_forward_steps
        )
        batched_frac = (
            self._engine.batched_forward_steps / total_forwards
            if total_forwards
            else 0.0
        )
        return {
            "avg_batch_size": round(stats.avg_batch_size, 3),
            "max_batch_size": stats.max_active,
            "batched_forward_frac": round(batched_frac, 3),
        }

    def count_tokens(self, text: str) -> int:
        if self._engine is None:
            return 0
        return self._engine.count_tokens(text)

    async def generate_stream(
        self, prompt: str, max_new_tokens: int
    ) -> AsyncIterator[str]:
        if self._engine is None:
            raise RuntimeError("backend not started")
        seq_id = await self._engine.submit(
            SubmitRequest(prompt=prompt, max_new_tokens=max_new_tokens)
        )
        async for ev in self._engine.stream(seq_id):
            if ev.done:
                break
            if ev.token_text:
                yield ev.token_text
        self._engine.retire(seq_id)

    def get_stats(self) -> dict:
        if self._engine is not None:
            return self._snapshot_stats()
        return self._final_stats
