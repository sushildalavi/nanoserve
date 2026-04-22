"""L1 parity: NanoServeEngine single-seq greedy output == HF model.generate()
greedy output, token-exact, across a small fixed prompt set.

this gate validates the manual forward-loop (engine.py). if it passes, the
scheduler + past_kv plumbing is producing the same tokens as transformers'
built-in generate path. L2 and L3 build on top of this.

test is skipped unless MPS is available — no point running it on CI that
doesn't have Apple Silicon.
"""
from __future__ import annotations

import asyncio

import pytest

from nanoserve.config import TINYLLAMA_HF
from nanoserve.engine.engine import NanoServeEngine
from nanoserve.engine.service import SubmitRequest

PROMPTS = [
    "What is 2 + 2?",
    "List three primary colors.",
    "Write the word hello.",
]
MAX_NEW = 16


def _mps_available() -> bool:
    try:
        import torch
        return torch.backends.mps.is_available()
    except ImportError:
        return False


@pytest.fixture(scope="module")
def model_and_tokenizer():
    if not _mps_available():
        pytest.skip("mps not available on this host")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TINYLLAMA_HF.path)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        TINYLLAMA_HF.path,
        dtype=torch.float16,
        attn_implementation="eager",
    ).to("mps")
    model.eval()
    model.generation_config.max_length = None
    return model, tok


def _hf_reference_output_ids(model, tok, prompt: str) -> list[int]:
    """run HF generate greedy and return only the newly-generated token ids."""
    import torch

    messages = [{"role": "user", "content": prompt}]
    templated = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tok(templated, return_tensors="pt").to("mps")
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    return out[0, prompt_len:].tolist()


async def _engine_output_ids(engine: NanoServeEngine, prompt: str) -> list[int]:
    req = SubmitRequest(prompt=prompt, max_new_tokens=MAX_NEW)
    seq_id = await engine.submit(req)
    async for _ in engine.stream(seq_id):
        pass
    seq = engine.get_finished_seq(seq_id)
    if seq is None:
        raise AssertionError("engine did not finish the sequence")
    engine.retire(seq_id)
    return seq.output_ids


@pytest.mark.parametrize("prompt", PROMPTS)
def test_l1_parity_per_prompt(model_and_tokenizer, prompt: str):
    """engine output tokens must match HF generate tokens exactly under greedy."""
    model, tok = model_and_tokenizer

    ref_ids = _hf_reference_output_ids(model, tok, prompt)

    engine = NanoServeEngine(TINYLLAMA_HF, max_batch_size=1, batching_mode="serial")

    async def run():
        engine._loop = asyncio.get_running_loop()
        engine._tokenizer = tok
        engine._model = model
        out_ids = await _engine_output_ids(engine, prompt)
        await engine.stop()
        return out_ids

    out_ids = asyncio.run(run())

    assert out_ids == ref_ids, (
        f"L1 parity failed for prompt {prompt!r}\n"
        f"  ref ({len(ref_ids)} toks): {ref_ids}\n"
        f"  eng ({len(out_ids)} toks): {out_ids}"
    )
