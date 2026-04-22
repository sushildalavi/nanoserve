from dataclasses import dataclass, field

from nanoserve.engine.sequence import Sequence, SeqStatus


@dataclass
class SchedulerConfig:
    max_batch_size: int = 8
    batching_mode: str = "continuous"  # "continuous" or "serial"

    def __post_init__(self):
        if self.batching_mode not in ("continuous", "serial"):
            raise ValueError(f"bad batching_mode: {self.batching_mode}")
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")


@dataclass
class SchedulerStats:
    """running counters the scheduler maintains. engine reads these at the end
    of a run to populate the ablation row.
    """
    steps_taken: int = 0
    sum_active_over_steps: int = 0
    max_active: int = 0

    @property
    def avg_batch_size(self) -> float:
        if self.steps_taken == 0:
            return 0.0
        return self.sum_active_over_steps / self.steps_taken


@dataclass
class Scheduler:
    """pure-python state machine. no tensors, no model deps.
    holds three queues (waiting / running / finished) and a tiny policy for
    which sequences get admitted each step.

    serial mode: at most 1 sequence runs at a time. new arrivals wait even if
    the batch has room. this is the flag-flipped ablation baseline.

    continuous mode: up to max_batch_size run concurrently, new arrivals
    admitted as soon as there's room (fcfs).
    """
    cfg: SchedulerConfig
    waiting: list[Sequence] = field(default_factory=list)
    running: list[Sequence] = field(default_factory=list)
    finished: list[Sequence] = field(default_factory=list)
    stats: SchedulerStats = field(default_factory=SchedulerStats)

    def submit(self, seq: Sequence) -> None:
        if seq.status != SeqStatus.WAITING:
            raise ValueError(f"submit expects WAITING, got {seq.status}")
        self.waiting.append(seq)

    def _room(self) -> int:
        if self.cfg.batching_mode == "serial":
            return max(0, 1 - len(self.running))
        return max(0, self.cfg.max_batch_size - len(self.running))

    def admit_ready(self) -> list[Sequence]:
        """move up to `room` waiting seqs into PREFILLING. returns the newly
        admitted seqs so the engine can run prefill on them.
        """
        room = self._room()
        admitted: list[Sequence] = []
        while room > 0 and self.waiting:
            seq = self.waiting.pop(0)
            seq.status = SeqStatus.PREFILLING
            admitted.append(seq)
            self.running.append(seq)
            room -= 1
        return admitted

    def mark_prefill_done(self, seq: Sequence) -> None:
        if seq.status != SeqStatus.PREFILLING:
            raise ValueError(f"mark_prefill_done expects PREFILLING, got {seq.status}")
        seq.status = SeqStatus.DECODING

    def pick_decode_batch(self) -> list[Sequence]:
        """return every currently DECODING sequence. engine runs one forward
        step across them.
        """
        batch = [s for s in self.running if s.status == SeqStatus.DECODING]
        self.stats.steps_taken += 1
        self.stats.sum_active_over_steps += len(batch)
        if len(batch) > self.stats.max_active:
            self.stats.max_active = len(batch)
        return batch

    def mark_finished(self, seq: Sequence, reason: str) -> None:
        if seq.status == SeqStatus.FINISHED:
            return
        seq.status = SeqStatus.FINISHED
        seq.stop_reason = reason
        try:
            self.running.remove(seq)
        except ValueError:
            pass
        self.finished.append(seq)

    def retire(self, seq: Sequence) -> None:
        """fully drop a finished seq from all queues. called after results
        have been drained.
        """
        try:
            self.finished.remove(seq)
        except ValueError:
            pass

    def has_pending_work(self) -> bool:
        return bool(self.waiting or self.running)
