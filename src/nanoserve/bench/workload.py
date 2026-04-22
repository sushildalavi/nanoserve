import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Request:
    idx: int
    prompt: str
    max_new_tokens: int
    arrival_offset_s: float


def load_prompts(path: Path) -> list[str]:
    prompts: list[str] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompts.append(obj["prompt"])
    if not prompts:
        raise ValueError(f"no prompts in {path}")
    return prompts


def poisson_arrivals(n: int, rate: float, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    offsets: list[float] = []
    t = 0.0
    for _ in range(n):
        t += rng.expovariate(rate)
        offsets.append(t)
    return offsets


def closed_loop_arrivals(n: int) -> list[float]:
    return [0.0] * n


def build_workload(
    prompts: list[str],
    kind: str,
    num_requests: int,
    rate: float,
    max_new_tokens: int,
    seed: int = 0,
) -> list[Request]:
    rng = random.Random(seed)
    if kind == "poisson":
        offsets = poisson_arrivals(num_requests, rate, seed=seed)
    elif kind in ("closed-loop", "single"):
        offsets = closed_loop_arrivals(num_requests)
    else:
        raise ValueError(f"unknown workload kind: {kind}")

    reqs = []
    for i, t in enumerate(offsets):
        p = rng.choice(prompts)
        reqs.append(Request(idx=i, prompt=p, max_new_tokens=max_new_tokens, arrival_offset_s=t))
    return reqs
