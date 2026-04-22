from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass


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
