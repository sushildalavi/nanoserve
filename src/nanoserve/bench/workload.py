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


SHARED_PREFIX_TEMPLATE = (
    "You are a careful, helpful assistant. Always answer concisely, with "
    "specific examples when relevant. Avoid disclaimers and unnecessary "
    "preamble. Reply only with what the user actually asked for and nothing "
    "more. Do not begin responses with phrases like 'sure' or 'as an AI'. "
    "If you do not know an answer, say so directly in one sentence. Treat "
    "every prompt as if it came from an experienced engineer. "
)


def _shared_prefix(num_words: int) -> str:
    """take the first num_words words of SHARED_PREFIX_TEMPLATE. lets the
    workload generator simulate a system-prompt-style shared prefix that
    every request begins with.
    """
    if num_words <= 0:
        return ""
    words = SHARED_PREFIX_TEMPLATE.split()
    if num_words >= len(words):
        return SHARED_PREFIX_TEMPLATE
    return " ".join(words[:num_words]) + " "


def build_workload(
    prompts: list[str],
    kind: str,
    num_requests: int,
    rate: float,
    max_new_tokens: int,
    seed: int = 0,
    shared_prefix_tokens: int = 0,
) -> list[Request]:
    rng = random.Random(seed)
    if kind == "poisson":
        offsets = poisson_arrivals(num_requests, rate, seed=seed)
    elif kind in ("closed-loop", "single"):
        offsets = closed_loop_arrivals(num_requests)
    else:
        raise ValueError(f"unknown workload kind: {kind}")

    prefix = _shared_prefix(shared_prefix_tokens)

    reqs = []
    for i, t in enumerate(offsets):
        p = rng.choice(prompts)
        reqs.append(
            Request(
                idx=i,
                prompt=prefix + p if prefix else p,
                max_new_tokens=max_new_tokens,
                arrival_offset_s=t,
            )
        )
    return reqs
