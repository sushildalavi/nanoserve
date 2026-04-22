"""NanoServeEngine — one engine, two modes (serial / continuous).

phase 2a scope: single-sequence manual prefill + decode loop using
model(input_ids, past_key_values=...) directly. no model.generate() calls.
this is the foundation L1/L2/L3 parity gates will run against.

the forward-pass code here is deliberately naive. all optimization (batching,
paged kv, masks) lands in subsequent commits with locked ablation rows.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

from nanoserve.config import ModelSpec
from nanoserve.engine.scheduler import Scheduler, SchedulerConfig
from nanoserve.engine.sequence import Sequence, next_seq_id
from nanoserve.engine.service import EngineService, SubmitRequest, TokenEvent


class NanoServeEngine(EngineService):
    """single-sequence forward loop. one request at a time, no batching yet.

    the batching_mode flag is carried so that downstream tests and ablation
    rows already distinguish serial (default today) from continuous (coming
    in day 4). a batching_mode of 'continuous' here currently still runs
    one-at-a-time — it just goes through the scheduler's multi-slot path
    so the plumbing is exercised.
    """

    def __init__(
        self,
        model_spec: ModelSpec,
        max_batch_size: int = 1,
        batching_mode: str = "serial",
    ):
        self.model_spec = model_spec
        self._device = "mps"
        self._tokenizer = None
        self._model = None
        self._scheduler = Scheduler(
            cfg=SchedulerConfig(
                max_batch_size=max_batch_size,
                batching_mode=batching_mode,
            )
        )
        self._streams: dict[int, asyncio.Queue] = {}
        self._lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    async def start(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.backends.mps.is_available():
            raise RuntimeError("mps not available")

        self._loop = asyncio.get_running_loop()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_spec.path)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_spec.path,
            dtype=torch.float16,
            attn_implementation="eager",
        ).to(self._device)
        self._model.eval()
        self._model.generation_config.max_length = None

    async def stop(self) -> None:
        self._model = None
        self._tokenizer = None
        for q in list(self._streams.values()):
            q.put_nowait(None)
        self._streams.clear()

    def count_tokens(self, text: str) -> int:
        if self._tokenizer is None or not text:
            return 0
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    async def submit(self, req: SubmitRequest) -> int:
        if self._tokenizer is None:
            raise RuntimeError("engine not started")

        messages = [{"role": "user", "content": req.prompt}]
        templated = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self._tokenizer.encode(templated, add_special_tokens=False)

        seq = Sequence(
            id=next_seq_id(),
            prompt_ids=prompt_ids,
            max_new_tokens=req.max_new_tokens,
            eos_token_id=(
                req.eos_token_id
                if req.eos_token_id is not None
                else self._tokenizer.eos_token_id
            ),
            submit_ts=time.time(),
        )
        self._streams[seq.id] = asyncio.Queue()
        self._scheduler.submit(seq)
        return seq.id

    async def stream(self, seq_id: int) -> AsyncIterator[TokenEvent]:
        if seq_id not in self._streams:
            raise KeyError(f"unknown seq_id: {seq_id}")
        q = self._streams[seq_id]

        # serial-lock the drain so concurrent callers don't interleave
        # forward passes on the same model.
        async with self._lock:
            driver = asyncio.create_task(asyncio.to_thread(self._drain_loop, seq_id))
            try:
                while True:
                    ev = await q.get()
                    if ev is None:
                        break
                    yield ev
                    if ev.done:
                        break
            finally:
                await driver
                self._streams.pop(seq_id, None)

    def get_finished_seq(self, seq_id: int) -> Sequence | None:
        """look up a finished sequence by id. used by callers (tests, runner)
        that need to read output_ids or timing after stream() completes.
        caller owns retiring it afterward via retire().
        """
        return next(
            (s for s in self._scheduler.finished if s.id == seq_id), None
        )

    def retire(self, seq_id: int) -> None:
        seq = self.get_finished_seq(seq_id)
        if seq is not None:
            self._scheduler.retire(seq)

    # ---- sync drain loop, runs on a worker thread ----

    def _drain_loop(self, seq_id: int) -> None:
        import torch

        sched = self._scheduler
        tok = self._tokenizer
        model = self._model
        device = self._device

        while sched.has_pending_work():
            for seq in sched.admit_ready():
                seq.admit_ts = time.time()
                self._prefill(seq, model, tok, device, torch)
                sched.mark_prefill_done(seq)

            batch = sched.pick_decode_batch()
            if not batch:
                continue

            for seq in batch:
                self._decode_one(seq, model, device, torch)
                stop = seq.should_stop()
                if stop is not None:
                    seq.finish_ts = time.time()
                    sched.mark_finished(seq, reason=stop)
                    self._emit_done(seq, stop)

            if any(s.id == seq_id for s in sched.finished):
                return

    def _prefill(self, seq: Sequence, model, tokenizer, device, torch_mod) -> None:
        input_ids = torch_mod.tensor([seq.prompt_ids], device=device)
        with torch_mod.inference_mode():
            out = model(input_ids=input_ids, use_cache=True)
        seq.past_kv = out.past_key_values
        next_tok = int(out.logits[0, -1].argmax().item())
        seq.append_token(next_tok)
        seq.first_token_ts = time.time()
        self._emit_token(seq, next_tok)

    def _decode_one(self, seq: Sequence, model, device, torch_mod) -> None:
        last_tok = seq.output_ids[-1]
        input_ids = torch_mod.tensor([[last_tok]], device=device)
        with torch_mod.inference_mode():
            out = model(
                input_ids=input_ids,
                past_key_values=seq.past_kv,
                use_cache=True,
            )
        seq.past_kv = out.past_key_values
        next_tok = int(out.logits[0, -1].argmax().item())
        seq.append_token(next_tok)
        self._emit_token(seq, next_tok)

    def _emit_token(self, seq: Sequence, tok_id: int) -> None:
        q = self._streams.get(seq.id)
        if q is None or self._loop is None:
            return
        text = self._tokenizer.decode([tok_id], skip_special_tokens=True)
        ev = TokenEvent(seq_id=seq.id, token_text=text, done=False)
        self._loop.call_soon_threadsafe(q.put_nowait, ev)

    def _emit_done(self, seq: Sequence, reason: str) -> None:
        q = self._streams.get(seq.id)
        if q is None or self._loop is None:
            return
        done_ev = TokenEvent(
            seq_id=seq.id, token_text="", done=True, stop_reason=reason
        )
        self._loop.call_soon_threadsafe(q.put_nowait, done_ev)
        self._loop.call_soon_threadsafe(q.put_nowait, None)
