from dataclasses import dataclass, field
from enum import StrEnum
from itertools import count
from typing import Any


class SeqStatus(StrEnum):
    WAITING = "waiting"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    FINISHED = "finished"


_id_counter = count(1)


def next_seq_id() -> int:
    return next(_id_counter)


@dataclass
class Sequence:
    id: int
    prompt_ids: list[int]
    max_new_tokens: int
    output_ids: list[int] = field(default_factory=list)
    status: SeqStatus = SeqStatus.WAITING
    past_kv: Any = None
    submit_ts: float = 0.0
    admit_ts: float = 0.0
    first_token_ts: float = 0.0
    finish_ts: float = 0.0
    eos_token_id: int | None = None
    stop_reason: str | None = None

    @property
    def all_ids(self) -> list[int]:
        return self.prompt_ids + self.output_ids

    @property
    def total_tokens(self) -> int:
        return len(self.prompt_ids) + len(self.output_ids)

    def append_token(self, tok: int) -> None:
        self.output_ids.append(tok)

    def should_stop(self) -> str | None:
        """return a stop reason if this sequence must finish, else None."""
        if len(self.output_ids) >= self.max_new_tokens:
            return "max_tokens"
        if (
            self.eos_token_id is not None
            and self.output_ids
            and self.output_ids[-1] == self.eos_token_id
        ):
            return "eos"
        return None
