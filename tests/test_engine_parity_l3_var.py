"""L3-var parity: batched forward with variable-length prompts must produce
identical outputs to running each prompt in isolation.

this is the 2B correctness gate. L3 (2A) proved batching works for same-
length prompts. L3-var proves left-padding + attention_mask don't corrupt
the shorter sequences. if this passes, the 2B Poisson sweep can trust
that heterogeneous batches are correct.
"""
from __future__ import annotations

import asyncio

import pytest

from nanoserve.config import TINYLLAMA_HF
from nanoserve.engine.engine import NanoServeEngine
from nanoserve.engine.service import SubmitRequest

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


def _pick_different_length_prompts(tok) -> list[str]:
    """select two prompts whose templated+tokenized lengths differ. if the
    lengths are equal we fall back to L3 — that's fine, but the whole point
    is to exercise masking, so we prefer a pair that actually differs.
    """
    candidates = [
        "What is 2 + 2?",
        "List three primary colors.",
        "Write the word hello.",
        "Say the word cat.",
        "Tell me a color.",
        "Name a fruit.",
        "What is pi?",
        "Say hi.",
        "Count to five please.",
    ]

    def _len(p: str) -> int:
        t = tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return len(tok.encode(t, add_special_tokens=False))

    sized = sorted(candidates, key=_len)
    # pick the shortest and something at least 2 tokens longer
    short = sized[0]
    short_len = _len(short)
    longer = next((p for p in sized if _len(p) >= short_len + 2), sized[-1])
    return [short, longer]


async def _isolated_output_ids(model, tok, prompt: str) -> list[int]:
    engine = NanoServeEngine(TINYLLAMA_HF, max_batch_size=1, batching_mode="serial")
    engine._loop = asyncio.get_running_loop()
    engine._tokenizer = tok
    engine._model = model
    engine._start_driver()
    try:
        sid = await engine.submit(SubmitRequest(prompt=prompt, max_new_tokens=MAX_NEW))
        async for _ in engine.stream(sid):
            pass
        seq = engine.get_finished_seq(sid)
        assert seq is not None
        engine.retire(sid)
        return seq.output_ids
    finally:
        await engine.stop()


async def _batched_outputs(model, tok, prompts: list[str]) -> list[list[int]]:
    engine = NanoServeEngine(
        TINYLLAMA_HF, max_batch_size=len(prompts), batching_mode="continuous"
    )
    engine._loop = asyncio.get_running_loop()
    engine._tokenizer = tok
    engine._model = model

    calls = {"prefill_batch": 0, "decode_batch": 0}
    orig_pb = engine._prefill_batch
    orig_db = engine._decode_batch

    def wrap_pb(seqs, tm):
        calls["prefill_batch"] += 1
        return orig_pb(seqs, tm)

    def wrap_db(batch, tm):
        calls["decode_batch"] += 1
        return orig_db(batch, tm)

    engine._prefill_batch = wrap_pb  # type: ignore[method-assign]
    engine._decode_batch = wrap_db  # type: ignore[method-assign]
    engine._start_driver()
    try:
        ids = await engine.submit_many(
            [SubmitRequest(prompt=p, max_new_tokens=MAX_NEW) for p in prompts]
        )

        async def drain(sid: int):
            async for _ in engine.stream(sid):
                pass

        await asyncio.gather(*(drain(sid) for sid in ids))

        outs = []
        for sid in ids:
            seq = engine.get_finished_seq(sid)
            assert seq is not None
            outs.append(seq.output_ids)
            engine.retire(sid)
        assert calls["prefill_batch"] == 1, (
            f"expected exactly one batched prefill call, got {calls['prefill_batch']}"
        )
        assert calls["decode_batch"] > 0, (
            "decode_batch never fired — test did not exercise the batched path"
        )
        return outs
    finally:
        await engine.stop()


def test_l3_var_parity(model_and_tokenizer):
    model, tok = model_and_tokenizer
    prompts = _pick_different_length_prompts(tok)

    async def run():
        refs = [await _isolated_output_ids(model, tok, p) for p in prompts]
        outs = await _batched_outputs(model, tok, prompts)
        return refs, outs

    refs, outs = asyncio.run(run())

    for p, ref, out in zip(prompts, refs, outs, strict=True):
        assert out == ref, (
            f"L3-var parity failed on prompt {p!r}\n"
            f"  ref ({len(ref)}): {ref}\n"
            f"  out ({len(out)}): {out}"
        )
