"""service-core interface used by runners, cli, and (later) fastapi.
the engine itself implements this; every caller talks through these methods
so swapping in a real scheduler later is a no-op at the api boundary.
"""
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol


@dataclass
class SubmitRequest:
    prompt: str
    max_new_tokens: int
    eos_token_id: int | None = None


@dataclass
class TokenEvent:
    seq_id: int
    token_text: str
    done: bool = False
    stop_reason: str | None = None


class EngineService(Protocol):
    """the minimal service surface an engine must expose.

    submit enqueues a request; stream yields token events until the sequence
    finishes. implementations are free to batch requests internally — the
    caller doesn't know and shouldn't care.
    """

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def submit(self, req: SubmitRequest) -> int: ...
    async def stream(self, seq_id: int) -> AsyncIterator[TokenEvent]: ...
    def count_tokens(self, text: str) -> int: ...
