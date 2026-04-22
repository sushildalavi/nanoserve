from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

import psutil


@dataclass
class GenOutput:
    text: str
    input_tokens: int
    output_tokens: int


class Backend(ABC):
    name: str = "base"

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    async def generate_stream(
        self, prompt: str, max_new_tokens: int
    ) -> AsyncIterator[str]:
        """yield token text as it is produced. implementation is an async generator."""
        ...

    @abstractmethod
    def count_tokens(self, text: str) -> int: ...

    def get_mem_mb(self) -> float:
        """resident memory attributable to this backend in mb.
        default assumes the backend lives in the current process (hf-mps).
        subclasses that spawn a subprocess should override.
        """
        return psutil.Process().memory_info().rss / (1024 * 1024)

    def get_stats(self) -> dict:
        """backend-specific counters (avg batch size, etc). default empty.
        the runner merges this into the ablation row.
        """
        return {}
