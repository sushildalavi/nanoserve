"""L3 parity: N sequences batched in a SINGLE forward step must produce the
same tokens as running each one in isolation.

the L2 gate validated interleaved scheduling with per-seq forwards. L3
validates the batched forward path specifically — stacking per-seq
past_kv caches, running one model.forward across them, splitting the
result back. if a row's tokens diverge from the isolated reference, the
bug is in the cache merge/split or the batched logits indexing.

2A precondition: all prompts must tokenize to the same length. this test
uses two hand-picked prompts that happen to template + tokenize to the
same length under TinyLlama's chat template.

the assertion of interest is that _can_batch_forward actually fires
during this run (otherwise L3 is really running L2 again). we inject a
tiny counter to verify the batched path was taken.
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


def _pick_same_length_prompts(tok) -> tuple[str, str]:
    """find two prompts whose templated+tokenized lengths match."""
    candidates = [
        "What is 2 + 2?",
        "List three primary colors.",
        "Write the word hello.",
        "Say the word cat.",
        "Tell me a color.",
        "Name a fruit.",
        "What is pi?",
        "Say hi to the user.",
    ]

    def _len(p: str) -> int:
        t = tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return len(tok.encode(t, add_special_tokens=False))

    by_len: dict[int, list[str]] = {}
    for p in candidates:
        by_len.setdefault(_len(p), []).append(p)

    for _, prompts in by_len.items():
        if len(prompts) >= 2:
            return prompts[0], prompts[1]
    raise AssertionError(f"no two prompts share a tokenized length: {by_len}")


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
    """run all prompts concurrently through a continuous-mode engine of
    size len(prompts). count how many times the batched path actually
    fired.
    """
    engine = NanoServeEngine(
        TINYLLAMA_HF, max_batch_size=len(prompts), batching_mode="continuous"
    )
    engine._loop = asyncio.get_running_loop()
    engine._tokenizer = tok
    engine._model = model

    calls = {"batched": 0}
    original = engine._decode_batch

    def counted(batch, torch_mod):
        calls["batched"] += 1
        return original(batch, torch_mod)

    engine._decode_batch = counted  # type: ignore[method-assign]
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
        assert calls["batched"] > 0, (
            "batched forward path was never taken — L3 did not exercise the "
            "code it claims to validate"
        )
        return outs
    finally:
        await engine.stop()


def test_l3_parity_batched_forward(model_and_tokenizer):
    model, tok = model_and_tokenizer
    p1, p2 = _pick_same_length_prompts(tok)

    async def run():
        ref1 = await _isolated_output_ids(model, tok, p1)
        ref2 = await _isolated_output_ids(model, tok, p2)
        out1, out2 = await _batched_outputs(model, tok, [p1, p2])
        return (p1, ref1, out1), (p2, ref2, out2)

    (p1, ref1, out1), (p2, ref2, out2) = asyncio.run(run())

    assert out1 == ref1, (
        f"L3 parity failed on prompt {p1!r}\n  ref: {ref1}\n  out: {out1}"
    )
    assert out2 == ref2, (
        f"L3 parity failed on prompt {p2!r}\n  ref: {ref2}\n  out: {out2}"
    )
