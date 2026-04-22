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
    ):
        self.model_spec = model
        self.batching_mode = batching_mode
        self.max_batch_size = max_batch_size
        self._engine: NanoServeEngine | None = None

    async def start(self) -> None:
        self._engine = NanoServeEngine(
            self.model_spec,
            max_batch_size=self.max_batch_size,
            batching_mode=self.batching_mode,
        )
        await self._engine.start()

    async def stop(self) -> None:
        if self._engine is not None:
            await self._engine.stop()
            self._engine = None

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
