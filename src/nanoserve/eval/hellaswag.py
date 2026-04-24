"""a small completion-quality eval.

if `datasets` is installed, this pulls `Rowan/hellaswag` validation and
scores the first N items. otherwise it falls back to a small built-in
cloze fixture that mirrors the HellaSwag shape (context + 4 endings +
correct label). the fixture is NOT a reproduction of HellaSwag — it is
an in-repo smoke test so the eval runs offline.

scoring: for each item, compute the mean negative log-likelihood of each
ending conditioned on the context, pick the lowest, and check against
the gold label. this is the standard LM-scoring recipe used by lm-
evaluation-harness.

TinyLlama at 1.1B parameters is expected to score in the mid-30s on real
HellaSwag (random chance is 25%). the eval is useful primarily for
RELATIVE comparison across quant modes — we expect int4 to drop a few
points below fp16, not to reach the published TinyLlama number.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

# cloze-style fixture. 12 items, 4 choices each, label = index of the
# correct ending. drawn from general everyday scenarios, not HellaSwag.
FIXTURE_ITEMS: list[dict] = [
    {
        "ctx": "A man is holding an umbrella in heavy rain. He",
        "endings": [
            "steps inside a cafe to stay dry.",
            "plants a tree in the garden.",
            "begins juggling oranges on the sidewalk.",
            "paints the clouds with a brush.",
        ],
        "label": 0,
    },
    {
        "ctx": "The chef cracked two eggs into a bowl. She",
        "endings": [
            "drove the bowl to the airport.",
            "whisked them together with a fork.",
            "sang a lullaby to the eggs.",
            "buried the bowl under the sink.",
        ],
        "label": 1,
    },
    {
        "ctx": "The toddler reached for the cookie jar. Her mother",
        "endings": [
            "launched a satellite into orbit.",
            "went skiing on a snowy mountain.",
            "moved the jar to a higher shelf.",
            "painted the jar with glitter.",
        ],
        "label": 2,
    },
    {
        "ctx": "The runner crossed the finish line first. She",
        "endings": [
            "raised her arms in celebration.",
            "fell asleep on the track.",
            "built a birdhouse out of paper.",
            "sent a fax to the mayor.",
        ],
        "label": 0,
    },
    {
        "ctx": "He forgot his wallet at home. At the coffee shop he",
        "endings": [
            "ordered a triple espresso anyway.",
            "flew a kite across the counter.",
            "asked to borrow money from a friend.",
            "built a sandcastle on the table.",
        ],
        "label": 2,
    },
    {
        "ctx": "The gardener planted tulip bulbs in autumn. By spring they",
        "endings": [
            "turned into small wooden chairs.",
            "bloomed in red and yellow flowers.",
            "vanished into a puff of smoke.",
            "became a flock of seagulls.",
        ],
        "label": 1,
    },
    {
        "ctx": "The cat jumped onto the kitchen counter. The owner",
        "endings": [
            "congratulated the cat with a medal.",
            "gently placed the cat back on the floor.",
            "handed the cat the car keys.",
            "painted stripes on the cat's tail.",
        ],
        "label": 1,
    },
    {
        "ctx": "A strong wind blew her hat off her head. She",
        "endings": [
            "planted corn in a nearby field.",
            "soldered a circuit board together.",
            "chased after it down the sidewalk.",
            "built a wooden pier out of sand.",
        ],
        "label": 2,
    },
    {
        "ctx": "He opened the oven and smelled something burning. He",
        "endings": [
            "started a choir of onions.",
            "quickly pulled the tray out to check.",
            "filed a patent for the smell.",
            "built a small boat from the ashes.",
        ],
        "label": 1,
    },
    {
        "ctx": "The library closes at nine. At eight fifty-five, the librarian",
        "endings": [
            "dimmed the lights and began locking up.",
            "launched a rocket to the moon.",
            "started a ten-hour karaoke session.",
            "painted the books with watercolors.",
        ],
        "label": 0,
    },
    {
        "ctx": "She spilled coffee on her laptop keyboard. She",
        "endings": [
            "scheduled a meeting in the spilled coffee.",
            "taught the laptop to swim.",
            "grabbed a towel and tried to dry it.",
            "rolled the laptop down a hill.",
        ],
        "label": 2,
    },
    {
        "ctx": "The toddler's balloon floated up into the sky. He",
        "endings": [
            "cried and pointed at the balloon.",
            "read a newspaper upside down.",
            "filed his income taxes early.",
            "baked a loaf of sourdough.",
        ],
        "label": 0,
    },
]


@dataclass
class HSItem:
    ctx: str
    endings: list[str]
    label: int


def load_items(prefer_hellaswag: bool = True, max_items: int = 100) -> list[HSItem]:
    """return a list of HSItems. tries HellaSwag validation first, falls
    back to the built-in cloze fixture if datasets is unavailable.
    """
    if prefer_hellaswag:
        try:
            from datasets import load_dataset  # type: ignore
            ds = load_dataset("Rowan/hellaswag", split="validation")
            items: list[HSItem] = []
            for row in ds:
                # hellaswag concatenates activity_label + ctx_a + ctx_b
                # when ctx_b is a sentence-starter. lm-eval-harness uses:
                # ctx = activity_label + ": " + ctx_a + " " + ctx_b.capitalize()
                al = (row.get("activity_label") or "").strip()
                ca = (row.get("ctx_a") or "").strip()
                cb = (row.get("ctx_b") or "").strip()
                if cb:
                    cb = cb[0].upper() + cb[1:]
                    ctx = f"{al}: {ca} {cb}" if al else f"{ca} {cb}"
                else:
                    ctx = f"{al}: {ca}" if al else ca
                endings = list(row["endings"])
                if len(endings) != 4:
                    continue
                label = int(row["label"])
                items.append(HSItem(ctx=ctx, endings=endings, label=label))
                if len(items) >= max_items:
                    break
            if items:
                return items
        except Exception:
            pass  # fall through

    return [HSItem(**d) for d in FIXTURE_ITEMS]


@torch.inference_mode()
def _score_ending_nll(model, tokenizer, ctx: str, ending: str, device) -> float:
    """mean token NLL of `ending` conditioned on `ctx`, in nats/token.

    tokenizes ctx and ending separately then concatenates the ids, so BPE
    merges across the ctx/ending boundary can't silently move the split
    point (which would happen if we tokenized `ctx + " " + ending` as one
    string and trusted `ctx_ids` to be a prefix of the full encoding).

    the leading-space convention for the ending ("roll a space into the
    first ending token") matches how lm-evaluation-harness scores
    completion tasks on llama-family tokenizers.
    """
    ctx_ids = tokenizer(ctx, return_tensors="pt", add_special_tokens=True).input_ids
    end_ids = tokenizer(
        " " + ending, return_tensors="pt", add_special_tokens=False
    ).input_ids
    if end_ids.shape[1] == 0:
        return float("inf")
    input_ids = torch.cat([ctx_ids, end_ids], dim=1).to(device)

    ctx_len = ctx_ids.shape[1]
    # mask out context positions so loss comes from the ending only.
    labels = input_ids.clone()
    labels[:, :ctx_len] = -100
    out = model(input_ids, labels=labels)
    # out.loss is mean NLL over (ending_len - 1 + 1 ctx handoff) shift
    # positions where label != -100. for ranking endings we only need the
    # mean, so return it directly. NaN guard for pathological endings.
    loss = out.loss.item()
    if not math.isfinite(loss):
        return float("inf")
    return loss


@torch.inference_mode()
def score_items(
    model: nn.Module,
    tokenizer,
    items: list[HSItem],
    device: torch.device | str,
) -> dict[str, float]:
    """return accuracy and item count for the given items list."""
    correct = 0
    for it in items:
        nlls = [
            _score_ending_nll(model, tokenizer, it.ctx, e, device) for e in it.endings
        ]
        pred = int(min(range(len(nlls)), key=lambda i: nlls[i]))
        if pred == it.label:
            correct += 1
    n = len(items)
    return {
        "accuracy": correct / n if n else float("nan"),
        "n": float(n),
    }
