"""perplexity on a fixed natural-text slice.

a straightforward language-model PPL: concat the corpus, slide a window
of `max_seq_len` tokens with `stride` overlap, compute CE loss on each
window (masking the overlap so each token contributes exactly once),
then PPL = exp(mean_loss).

the corpus loader tries huggingface `datasets` (wikitext-2-raw-v1,
validation) first so you can compare against published numbers; if
`datasets` is not installed or the download fails, it falls back to the
local fixture at `prompts/eval/ppl_fixture.txt` — ~40 sentences of
general technical text. the absolute number differs between the two
paths but the RELATIVE comparison across quant modes (fp16 vs int8 vs
int4) is what the eval is designed to surface, and that works on either
corpus.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from nanoserve.config import REPO_ROOT

FIXTURE_PATH = REPO_ROOT / "prompts" / "eval" / "ppl_fixture.txt"


def load_corpus(prefer_wikitext: bool = True, max_chars: int = 80_000) -> tuple[str, str]:
    """return (text, source) where source is "wikitext-2-val-slice" if
    huggingface datasets provided wikitext, or "fixture" if we fell back
    to `prompts/eval/ppl_fixture.txt`. callers that just need the text
    can unpack and ignore source; the source is surfaced in eval.csv so
    runs against different corpora don't get compared as if they were
    the same benchmark.
    """
    if prefer_wikitext:
        try:
            from datasets import load_dataset  # type: ignore
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
            chunks: list[str] = []
            total = 0
            for row in ds:
                t = row["text"]
                if not t or not t.strip():
                    continue
                chunks.append(t)
                total += len(t)
                if total >= max_chars:
                    break
            if chunks:
                return ("\n".join(chunks)[:max_chars], "wikitext-2-val-slice")
        except Exception:
            pass  # fall through to fixture

    text = FIXTURE_PATH.read_text(encoding="utf-8")
    return (text, "fixture")


@torch.inference_mode()
def compute_perplexity(
    model: nn.Module,
    tokenizer,
    text: str,
    device: torch.device | str,
    max_seq_len: int = 512,
    stride: int = 256,
) -> dict[str, float]:
    """compute PPL with a sliding window, returning a dict suitable for
    the eval CSV row.

    - max_seq_len bounds the context; too large blows up the attention
      memory at no quality gain on TinyLlama.
    - stride controls window overlap; tokens in the overlap region are
      masked so each one contributes exactly once to the average loss.
    """
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    total_tokens = input_ids.shape[1]

    nlls: list[torch.Tensor] = []
    counted_tokens = 0
    prev_end = 0
    pos = 0
    while pos < total_tokens:
        end = min(pos + max_seq_len, total_tokens)
        window = input_ids[:, pos:end]
        target = window.clone()

        # mask tokens that were already counted in the previous window
        overlap = max(0, prev_end - pos)
        if overlap > 0:
            target[:, :overlap] = -100

        out = model(window, labels=target)
        # HF causal-LM loss shifts labels internally: shift_labels = labels[1:].
        # the averaging denominator is the count of shift positions where the
        # shifted label is not -100. work that out directly instead of
        # approximating from the unshifted mask:
        #   - shift_labels has (N-1) positions (0..N-2), corresponding to labels[1..N-1]
        #   - labels[i] == -100 for i in [0, overlap) -> shift positions [0, overlap-1) are ignored
        #   - so valid count = (N - 1) - max(0, overlap - 1)
        #                    = N - overlap   if overlap >= 1
        #                    = N - 1         if overlap == 0
        total_positions = target.shape[-1]
        if overlap == 0:
            valid_for_mean = total_positions - 1
        else:
            valid_for_mean = total_positions - overlap
        if valid_for_mean > 0 and torch.isfinite(out.loss):
            nlls.append(out.loss.detach().float() * valid_for_mean)
            counted_tokens += valid_for_mean

        prev_end = end
        if end == total_tokens:
            break
        pos += stride

    if counted_tokens == 0:
        return {"ppl": float("nan"), "nll": float("nan"), "tokens": 0}

    mean_nll = (torch.stack(nlls).sum() / counted_tokens).item()
    return {
        "ppl": math.exp(mean_nll),
        "nll": mean_nll,
        "tokens": counted_tokens,
    }


def load_model(quant_mode: str, model_path: str, device: torch.device | str):
    """load a tinyllama model and apply the requested quant path.
    returns (model, tokenizer).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    if quant_mode == "int8":
        from nanoserve.engine.quant import quantize_model_int8_weight_only
        quantize_model_int8_weight_only(model)
    elif quant_mode == "int4":
        from nanoserve.engine.quant_int4 import quantize_model_int4_weight_only
        quantize_model_int4_weight_only(model)
    elif quant_mode == "torchao_int8":
        from torchao.quantization import int8_weight_only, quantize_
        quantize_(model, int8_weight_only())
    elif quant_mode not in ("fp16", "none"):
        raise ValueError(f"unknown quant_mode: {quant_mode}")

    return model, tok
