"""MLX counterparts of `perplexity.compute_perplexity` and
`hellaswag.score_items`.

the main eval module targets pytorch models (HF causal LM). MLX uses a
different tensor API and different model-call convention (no `labels=`
kwarg — the loss is computed by hand against shifted logits), so the
cleanest structure is a parallel MLX path instead of an adapter.

output format matches the pytorch path so the runner can append a row
to `results/eval.csv` with source=`mlx-fp16` or `mlx-int4`.
"""
from __future__ import annotations

import math

from nanoserve.eval.hellaswag import HSItem


def load_mlx_model(quant_mode: str, model_path: str):
    """return (model, tokenizer) with optional int4/int8 quant applied.

    quant_mode values accepted here:
      - "fp16-mlx" / "mlx-fp16" / "fp16"  -> no quantization
      - "int4-mlx" / "mlx-int4" / "int4"  -> mlx.nn.quantize(bits=4)
      - "int8-mlx" / "mlx-int8" / "int8"  -> mlx.nn.quantize(bits=8)
    """
    from mlx.nn import quantize as mx_quantize
    from mlx_lm import load

    model, tokenizer = load(model_path)
    norm = quant_mode.replace("mlx-", "").replace("-mlx", "")
    if norm in ("int4", "int8"):
        bits = 4 if norm == "int4" else 8
        mx_quantize(model, group_size=64, bits=bits)
    elif norm not in ("fp16", "none"):
        raise ValueError(f"mlx eval: unknown quant_mode {quant_mode!r}")
    return model, tokenizer


def _cross_entropy_mean(logits, targets, mask):
    """mean cross entropy over positions where mask == True.

    logits: [1, L, V], targets: [1, L], mask: [1, L]. returns a python
    float in nats/token. uses log_softmax + gather to avoid building a
    full one-hot.
    """
    import mlx.core as mx
    import mlx.nn as nn

    # log-softmax over vocab; cast up to fp32 for numerical stability
    log_probs = nn.log_softmax(logits.astype(mx.float32), axis=-1)
    # gather the log-prob of the target at each position: [1, L]
    idx = mx.expand_dims(targets, axis=-1)  # [1, L, 1]
    picked = mx.take_along_axis(log_probs, idx, axis=-1).squeeze(axis=-1)  # [1, L]
    nll = -picked  # [1, L]
    mask_f = mask.astype(mx.float32)
    denom = mx.maximum(mask_f.sum(), mx.array(1.0))
    return float((nll * mask_f).sum() / denom)


def compute_perplexity_mlx(
    model,
    tokenizer,
    text: str,
    max_seq_len: int = 512,
    stride: int = 256,
) -> dict:
    """sliding-window PPL on MLX. mirrors the pytorch path: each token
    contributes to the averaged loss exactly once across windows.
    """
    import mlx.core as mx

    ids = tokenizer.encode(text) if hasattr(tokenizer, "encode") else tokenizer(text).input_ids
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    ids = list(ids)
    total = len(ids)
    if total < 2:
        return {"ppl": float("nan"), "nll": float("nan"), "tokens": 0}

    total_nll = 0.0
    counted = 0
    prev_end = 0
    pos = 0
    while pos < total:
        end = min(pos + max_seq_len, total)
        window = mx.array([ids[pos:end]])  # [1, N]
        # shift: predict window[:, 1:] from logits at window[:, :-1]
        logits = model(window)  # [1, N, V]
        shift_logits = logits[:, :-1, :]  # [1, N-1, V]
        shift_labels = window[:, 1:]       # [1, N-1]

        overlap = max(0, prev_end - pos)
        # mask out tokens that were already counted in the previous window.
        # the -1 in the overlap-shift math matches the shift of the labels.
        seq_len = shift_labels.shape[1]
        if overlap > 0:
            skip = max(0, overlap - 1)
        else:
            skip = 0
        mask_list = [0.0] * min(skip, seq_len) + [1.0] * max(0, seq_len - skip)
        mask = mx.array([mask_list[:seq_len]])

        valid_count = int(mask.sum())
        if valid_count > 0:
            mean_nll = _cross_entropy_mean(shift_logits, shift_labels, mask > 0.5)
            total_nll += mean_nll * valid_count
            counted += valid_count

        prev_end = end
        if end == total:
            break
        pos += stride

    if counted == 0:
        return {"ppl": float("nan"), "nll": float("nan"), "tokens": 0}
    mean = total_nll / counted
    return {"ppl": math.exp(mean), "nll": mean, "tokens": counted}


def _score_ending_nll_mlx(model, tokenizer, ctx: str, ending: str) -> float:
    """mean NLL of ending conditioned on ctx, in nats/token, on MLX."""
    import mlx.core as mx

    ctx_ids = tokenizer.encode(ctx) if hasattr(tokenizer, "encode") else tokenizer(ctx).input_ids
    end_ids = (
        tokenizer.encode(" " + ending, add_special_tokens=False)
        if hasattr(tokenizer, "encode")
        else tokenizer(" " + ending, add_special_tokens=False).input_ids
    )
    ctx_ids = list(ctx_ids.tolist() if hasattr(ctx_ids, "tolist") else ctx_ids)
    end_ids = list(end_ids.tolist() if hasattr(end_ids, "tolist") else end_ids)
    if len(end_ids) == 0:
        return float("inf")

    full = ctx_ids + end_ids
    input_ids = mx.array([full])  # [1, L]
    logits = model(input_ids)      # [1, L, V]
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    seq_len = shift_labels.shape[1]

    # mask: keep positions whose predicted TOKEN is an ending token.
    # shift_labels[i] is the token at position i+1 in the full sequence.
    # ending tokens start at index len(ctx_ids).
    ctx_len = len(ctx_ids)
    first_end_shift_idx = max(0, ctx_len - 1)
    mask_list = [
        1.0 if i >= first_end_shift_idx else 0.0 for i in range(seq_len)
    ]
    mask = mx.array([mask_list])
    if float(mask.sum()) == 0:
        return float("inf")
    return _cross_entropy_mean(shift_logits, shift_labels, mask > 0.5)


def score_items_mlx(model, tokenizer, items: list[HSItem]) -> dict:
    correct = 0
    for it in items:
        nlls = [_score_ending_nll_mlx(model, tokenizer, it.ctx, e) for e in it.endings]
        pred = int(min(range(len(nlls)), key=lambda i: nlls[i]))
        if pred == it.label:
            correct += 1
    n = len(items)
    return {"accuracy": correct / n if n else float("nan"), "n": float(n)}
