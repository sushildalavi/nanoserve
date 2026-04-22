import pytest

from nanoserve.bench.metrics import RequestRecord, aggregate, percentile


def test_percentile_single_value():
    assert percentile([42.0], 50) == 42.0
    assert percentile([42.0], 99) == 42.0


def test_percentile_empty():
    assert percentile([], 50) == 0.0


def test_percentile_known():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(xs, 0) == 1.0
    assert percentile(xs, 100) == 5.0
    assert percentile(xs, 50) == 3.0
    assert percentile(xs, 25) == 2.0


def test_percentile_bad_p():
    with pytest.raises(ValueError):
        percentile([1.0, 2.0], 150)


def _rec(idx, arrival, start, ft, end, out_t=8):
    return RequestRecord(
        idx=idx,
        arrival_ts=arrival,
        start_ts=start,
        first_token_ts=ft,
        end_ts=end,
        input_tokens=10,
        output_tokens=out_t,
    )


def test_record_derived_ms():
    r = _rec(0, 0.0, 0.1, 0.3, 1.1, out_t=9)
    assert r.queue_ms == pytest.approx(100.0)
    assert r.ttft_ms == pytest.approx(200.0)
    assert r.e2e_ms == pytest.approx(1100.0)
    assert r.decode_ms == pytest.approx(800.0)
    assert r.tpot_ms == pytest.approx(100.0)


def test_aggregate_basic():
    recs = [_rec(i, 0.0, 0.0, 0.1, 0.5, out_t=5) for i in range(10)]
    agg = aggregate(recs, wall_s=2.0)
    assert agg.n == 10
    assert agg.n_ok == 10
    assert agg.rps == pytest.approx(5.0)
    assert agg.output_tokens_total == 50
    assert agg.ttft_p50 == pytest.approx(100.0)


def test_aggregate_drops_failed_from_percentiles():
    good = [_rec(i, 0.0, 0.0, 0.1, 0.5) for i in range(5)]
    bad = RequestRecord(
        idx=99, arrival_ts=0.0, start_ts=0.0, first_token_ts=0.0, end_ts=0.0,
        input_tokens=0, output_tokens=0, ok=False, error="boom",
    )
    agg = aggregate(good + [bad], wall_s=1.0)
    assert agg.n == 6
    assert agg.n_ok == 5
