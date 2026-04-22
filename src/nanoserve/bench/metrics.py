from dataclasses import asdict, dataclass
from statistics import mean


@dataclass
class RequestRecord:
    idx: int
    arrival_ts: float
    start_ts: float
    first_token_ts: float
    end_ts: float
    input_tokens: int
    output_tokens: int
    ok: bool = True
    error: str | None = None

    @property
    def queue_ms(self) -> float:
        return (self.start_ts - self.arrival_ts) * 1000.0

    @property
    def ttft_ms(self) -> float:
        return (self.first_token_ts - self.start_ts) * 1000.0

    @property
    def e2e_ms(self) -> float:
        return (self.end_ts - self.arrival_ts) * 1000.0

    @property
    def decode_ms(self) -> float:
        return (self.end_ts - self.first_token_ts) * 1000.0

    @property
    def tpot_ms(self) -> float:
        n = max(self.output_tokens - 1, 1)
        return self.decode_ms / n


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if not 0.0 <= p <= 100.0:
        raise ValueError("percentile must be in [0, 100]")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    frac = k - lo
    return xs[lo] + (xs[hi] - xs[lo]) * frac


@dataclass
class AggregateMetrics:
    n: int
    n_ok: int
    duration_s: float
    rps: float
    input_tokens_total: int
    output_tokens_total: int
    decode_tok_s: float
    ttft_p50: float
    ttft_p95: float
    ttft_p99: float
    tpot_p50: float
    tpot_p95: float
    e2e_p50: float
    e2e_p95: float
    e2e_p99: float
    queue_p95: float

    def as_dict(self) -> dict:
        return asdict(self)


def aggregate(records: list[RequestRecord], wall_s: float) -> AggregateMetrics:
    ok = [r for r in records if r.ok]
    if not ok:
        raise ValueError("no successful records to aggregate")

    ttft = [r.ttft_ms for r in ok]
    tpot = [r.tpot_ms for r in ok]
    e2e = [r.e2e_ms for r in ok]
    queue = [r.queue_ms for r in ok]
    out_tokens = sum(r.output_tokens for r in ok)
    in_tokens = sum(r.input_tokens for r in ok)
    decode_seconds = sum((r.end_ts - r.first_token_ts) for r in ok)
    decode_tok_s = out_tokens / decode_seconds if decode_seconds > 0 else 0.0

    return AggregateMetrics(
        n=len(records),
        n_ok=len(ok),
        duration_s=wall_s,
        rps=len(ok) / wall_s if wall_s > 0 else 0.0,
        input_tokens_total=in_tokens,
        output_tokens_total=out_tokens,
        decode_tok_s=decode_tok_s,
        ttft_p50=percentile(ttft, 50),
        ttft_p95=percentile(ttft, 95),
        ttft_p99=percentile(ttft, 99),
        tpot_p50=percentile(tpot, 50),
        tpot_p95=percentile(tpot, 95),
        e2e_p50=percentile(e2e, 50),
        e2e_p95=percentile(e2e, 95),
        e2e_p99=percentile(e2e, 99),
        queue_p95=percentile(queue, 95),
    )


def avg_if(values: list[float]) -> float:
    return mean(values) if values else 0.0
