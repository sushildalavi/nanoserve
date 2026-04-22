import pytest

from nanoserve.engine.scheduler import Scheduler, SchedulerConfig
from nanoserve.engine.sequence import SeqStatus, Sequence, next_seq_id


def _seq(prompt_len: int = 4, max_new: int = 8) -> Sequence:
    return Sequence(
        id=next_seq_id(),
        prompt_ids=list(range(prompt_len)),
        max_new_tokens=max_new,
    )


def test_submit_puts_seq_in_waiting():
    s = Scheduler(cfg=SchedulerConfig(max_batch_size=4))
    seq = _seq()
    s.submit(seq)
    assert seq in s.waiting
    assert seq.status == SeqStatus.WAITING


def test_submit_rejects_non_waiting():
    s = Scheduler(cfg=SchedulerConfig())
    seq = _seq()
    seq.status = SeqStatus.DECODING
    with pytest.raises(ValueError):
        s.submit(seq)


def test_continuous_admits_up_to_max_batch():
    s = Scheduler(cfg=SchedulerConfig(max_batch_size=3))
    seqs = [_seq() for _ in range(5)]
    for q in seqs:
        s.submit(q)
    admitted = s.admit_ready()
    assert len(admitted) == 3
    assert len(s.waiting) == 2
    assert all(a.status == SeqStatus.PREFILLING for a in admitted)
    assert all(a in s.running for a in admitted)


def test_serial_admits_only_one_even_with_many_waiting():
    s = Scheduler(cfg=SchedulerConfig(batching_mode="serial", max_batch_size=8))
    for _ in range(5):
        s.submit(_seq())
    admitted = s.admit_ready()
    assert len(admitted) == 1
    assert len(s.running) == 1
    assert len(s.waiting) == 4


def test_serial_does_not_admit_while_one_is_running():
    s = Scheduler(cfg=SchedulerConfig(batching_mode="serial"))
    s.submit(_seq())
    s.submit(_seq())
    first = s.admit_ready()
    assert len(first) == 1
    again = s.admit_ready()
    assert again == []
    assert len(s.waiting) == 1


def test_prefill_transition_to_decoding():
    s = Scheduler(cfg=SchedulerConfig())
    seq = _seq()
    s.submit(seq)
    s.admit_ready()
    assert seq.status == SeqStatus.PREFILLING
    s.mark_prefill_done(seq)
    assert seq.status == SeqStatus.DECODING


def test_prefill_done_rejects_wrong_state():
    s = Scheduler(cfg=SchedulerConfig())
    seq = _seq()
    with pytest.raises(ValueError):
        s.mark_prefill_done(seq)


def test_pick_decode_batch_only_returns_decoding():
    s = Scheduler(cfg=SchedulerConfig(max_batch_size=4))
    seqs = [_seq() for _ in range(3)]
    for q in seqs:
        s.submit(q)
    admitted = s.admit_ready()
    assert s.pick_decode_batch() == []  # still PREFILLING
    for a in admitted:
        s.mark_prefill_done(a)
    batch = s.pick_decode_batch()
    assert len(batch) == 3


def test_stats_track_avg_and_max_batch():
    s = Scheduler(cfg=SchedulerConfig(max_batch_size=4))
    for _ in range(3):
        q = _seq()
        s.submit(q)
        for a in s.admit_ready():
            s.mark_prefill_done(a)
        s.pick_decode_batch()

    assert s.stats.steps_taken == 3
    assert s.stats.max_active == 3
    assert s.stats.avg_batch_size == pytest.approx(2.0)


def test_mark_finished_moves_seq_out_of_running():
    s = Scheduler(cfg=SchedulerConfig())
    seq = _seq()
    s.submit(seq)
    s.admit_ready()
    s.mark_prefill_done(seq)
    s.mark_finished(seq, reason="eos")
    assert seq.status == SeqStatus.FINISHED
    assert seq.stop_reason == "eos"
    assert seq not in s.running
    assert seq in s.finished


def test_retire_drops_from_finished():
    s = Scheduler(cfg=SchedulerConfig())
    seq = _seq()
    s.submit(seq)
    s.admit_ready()
    s.mark_prefill_done(seq)
    s.mark_finished(seq, reason="max_tokens")
    s.retire(seq)
    assert seq not in s.finished


def test_finish_frees_room_for_waiting():
    s = Scheduler(cfg=SchedulerConfig(max_batch_size=2))
    a, b, c = _seq(), _seq(), _seq()
    for q in (a, b, c):
        s.submit(q)
    first_round = s.admit_ready()
    assert {x.id for x in first_round} == {a.id, b.id}
    assert c in s.waiting
    s.mark_prefill_done(a)
    s.mark_finished(a, reason="max_tokens")
    second_round = s.admit_ready()
    assert second_round == [c]
    assert c in s.running


def test_has_pending_work():
    s = Scheduler(cfg=SchedulerConfig())
    assert not s.has_pending_work()
    seq = _seq()
    s.submit(seq)
    assert s.has_pending_work()
    s.admit_ready()
    assert s.has_pending_work()
    s.mark_prefill_done(seq)
    assert s.has_pending_work()
    s.mark_finished(seq, reason="eos")
    # finished-but-not-retired means results wait for the client to drain;
    # the engine itself has no more work, so this returns False
    assert not s.has_pending_work()
    s.retire(seq)
    assert not s.has_pending_work()


def test_sequence_should_stop_on_max_tokens():
    seq = _seq(max_new=3)
    seq.output_ids = [1, 2, 3]
    assert seq.should_stop() == "max_tokens"


def test_sequence_should_stop_on_eos():
    seq = _seq(max_new=100)
    seq.eos_token_id = 2
    seq.output_ids = [1, 2]
    assert seq.should_stop() == "eos"


def test_sequence_does_not_stop_early():
    seq = _seq(max_new=10)
    seq.output_ids = [1, 2, 3]
    assert seq.should_stop() is None


def test_bad_batching_mode_rejected():
    with pytest.raises(ValueError):
        SchedulerConfig(batching_mode="lol")


def test_bad_max_batch_rejected():
    with pytest.raises(ValueError):
        SchedulerConfig(max_batch_size=0)


def test_bad_admission_policy_rejected():
    with pytest.raises(ValueError):
        SchedulerConfig(admission_policy="lol")


def test_synchronized_policy_blocks_admit_while_running():
    s = Scheduler(
        cfg=SchedulerConfig(
            max_batch_size=4, admission_policy="synchronized"
        )
    )
    a, b, c = _seq(), _seq(), _seq()
    for q in (a, b, c):
        s.submit(q)
    first = s.admit_ready()
    assert {x.id for x in first} == {a.id, b.id, c.id}
    s.mark_prefill_done(a)
    s.mark_finished(a, reason="max_tokens")
    s.submit(_seq())
    blocked = s.admit_ready()
    assert blocked == [], (
        "synchronized policy must block admit while any seq is still running"
    )


def test_synchronized_policy_admits_after_drain():
    s = Scheduler(
        cfg=SchedulerConfig(
            max_batch_size=2, admission_policy="synchronized"
        )
    )
    a, b = _seq(), _seq()
    for q in (a, b):
        s.submit(q)
    first = s.admit_ready()
    assert len(first) == 2
    for x in first:
        s.mark_prefill_done(x)
        s.mark_finished(x, reason="max_tokens")
    c, d = _seq(), _seq()
    s.submit(c)
    s.submit(d)
    second = s.admit_ready()
    assert {x.id for x in second} == {c.id, d.id}


def test_fcfs_policy_admits_into_running_slots():
    s = Scheduler(
        cfg=SchedulerConfig(
            max_batch_size=4, admission_policy="fcfs"
        )
    )
    a = _seq()
    s.submit(a)
    s.admit_ready()
    s.mark_prefill_done(a)
    # a is still running. fcfs should still admit b into the spare slot.
    b = _seq()
    s.submit(b)
    second = s.admit_ready()
    assert second == [b]
