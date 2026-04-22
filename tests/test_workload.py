from nanoserve.bench.workload import build_workload, poisson_arrivals


def test_poisson_offsets_are_monotonic():
    xs = poisson_arrivals(n=50, rate=5.0, seed=7)
    assert len(xs) == 50
    for a, b in zip(xs, xs[1:]):
        assert b >= a


def test_poisson_is_deterministic_with_seed():
    a = poisson_arrivals(n=20, rate=2.0, seed=42)
    b = poisson_arrivals(n=20, rate=2.0, seed=42)
    assert a == b


def test_build_workload_closed_loop_zero_offsets():
    reqs = build_workload(
        prompts=["hi", "hello"],
        kind="closed-loop",
        num_requests=10,
        rate=1.0,
        max_new_tokens=8,
    )
    assert len(reqs) == 10
    assert all(r.arrival_offset_s == 0.0 for r in reqs)
    assert all(r.max_new_tokens == 8 for r in reqs)


def test_build_workload_rejects_unknown_kind():
    import pytest

    with pytest.raises(ValueError):
        build_workload(
            prompts=["hi"], kind="nope", num_requests=1, rate=1.0, max_new_tokens=1,
        )
