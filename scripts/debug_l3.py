"""debug script: run two concurrent requests through the engine with verbose
prints to find where L3 hangs.
"""
import asyncio
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nanoserve.config import TINYLLAMA_HF
from nanoserve.engine.engine import NanoServeEngine
from nanoserve.engine.service import SubmitRequest


def log(msg: str) -> None:
    print(f"[{time.time():.3f}] {msg}", flush=True)


async def main():
    log("loading tokenizer")
    tok = AutoTokenizer.from_pretrained(TINYLLAMA_HF.path)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    log("loading model")
    model = AutoModelForCausalLM.from_pretrained(
        TINYLLAMA_HF.path, dtype=torch.float16, attn_implementation="eager"
    ).to("mps")
    model.eval()
    model.generation_config.max_length = None
    log("model ready")

    engine = NanoServeEngine(TINYLLAMA_HF, max_batch_size=2, batching_mode="continuous")
    engine._loop = asyncio.get_running_loop()
    engine._tokenizer = tok
    engine._model = model

    cb = {"n": 0}
    one = {"n": 0}
    orig_cb = engine._decode_batch
    orig_one = engine._decode_one

    def wrap_cb(batch, tm):
        cb["n"] += 1
        log(f"  _decode_batch({len(batch)})")
        return orig_cb(batch, tm)

    def wrap_one(seq, tm):
        one["n"] += 1
        log(f"  _decode_one({seq.id})")
        return orig_one(seq, tm)

    engine._decode_batch = wrap_cb
    engine._decode_one = wrap_one
    engine._start_driver()
    log("driver started")

    log("submit_many")
    ids = await engine.submit_many(
        [
            SubmitRequest(prompt="What is pi?", max_new_tokens=8),
            SubmitRequest(prompt="Name a fruit.", max_new_tokens=8),
        ]
    )
    log(f"ids = {ids}")

    async def drain(sid):
        log(f"drain start sid={sid}")
        n = 0
        async for ev in engine.stream(sid):
            n += 1
            if ev.done:
                log(f"  sid={sid} DONE reason={ev.stop_reason}")
            else:
                log(f"  sid={sid} tok#{n}={ev.token_text!r}")
        log(f"drain exit sid={sid}")

    await asyncio.gather(*(drain(s) for s in ids))
    log(f"cb_calls={cb['n']} one_calls={one['n']}")
    await engine.stop()
    log("stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
