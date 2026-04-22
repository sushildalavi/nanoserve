"""NanoServeEngine — one engine, two modes (serial / continuous).

architecture: one persistent background thread drives the scheduler. the
event loop submits requests and consumes per-seq token queues. this lets
multiple callers stream concurrently — which is what L2 parity validates.

day 3 scope: the driver interleaves sequences in the scheduler but still
runs decode on one seq at a time inside each step. day 4 replaces the
per-step inner loop with a batched forward pass.
"""
from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator

from nanoserve.config import ModelSpec
from nanoserve.engine.scheduler import Scheduler, SchedulerConfig
from nanoserve.engine.sequence import Sequence, next_seq_id
from nanoserve.engine.service import EngineService, SubmitRequest, TokenEvent


class NanoServeEngine(EngineService):
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
        self._sched_lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._driver: threading.Thread | None = None
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

        self._start_driver()

    def _start_driver(self) -> None:
        self._stop.clear()
        self._driver = threading.Thread(target=self._driver_loop, daemon=True)
        self._driver.start()

    async def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._driver is not None:
            self._driver.join(timeout=5.0)
            self._driver = None
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
        with self._sched_lock:
            self._scheduler.submit(seq)
        self._wake.set()
        return seq.id

    async def stream(self, seq_id: int) -> AsyncIterator[TokenEvent]:
        if seq_id not in self._streams:
            raise KeyError(f"unknown seq_id: {seq_id}")
        q = self._streams[seq_id]
        try:
            while True:
                ev = await q.get()
                if ev is None:
                    break
                yield ev
                if ev.done:
                    break
        finally:
            self._streams.pop(seq_id, None)

    def get_finished_seq(self, seq_id: int) -> Sequence | None:
        with self._sched_lock:
            return next(
                (s for s in self._scheduler.finished if s.id == seq_id), None
            )

    def retire(self, seq_id: int) -> None:
        with self._sched_lock:
            seq = next(
                (s for s in self._scheduler.finished if s.id == seq_id), None
            )
            if seq is not None:
                self._scheduler.retire(seq)

    # ---- driver thread ----

    def _driver_loop(self) -> None:
        import torch

        while not self._stop.is_set():
            if not self._has_pending_work():
                self._wake.wait(timeout=1.0)
                self._wake.clear()
                continue

            # prefill any newly admitted sequences
            with self._sched_lock:
                admitted = self._scheduler.admit_ready()
            for seq in admitted:
                seq.admit_ts = time.time()
                self._prefill(seq, torch)
                with self._sched_lock:
                    self._scheduler.mark_prefill_done(seq)

            # step every currently-decoding sequence. day 3: one at a time.
            with self._sched_lock:
                batch = self._scheduler.pick_decode_batch()
            if not batch:
                continue
            for seq in batch:
                self._decode_one(seq, torch)
                stop = seq.should_stop()
                if stop is not None:
                    seq.finish_ts = time.time()
                    with self._sched_lock:
                        self._scheduler.mark_finished(seq, reason=stop)
                    self._emit_done(seq, stop)

    def _has_pending_work(self) -> bool:
        with self._sched_lock:
            return self._scheduler.has_pending_work()

    def _prefill(self, seq: Sequence, torch_mod) -> None:
        input_ids = torch_mod.tensor([seq.prompt_ids], device=self._device)
        with torch_mod.inference_mode():
            out = self._model(input_ids=input_ids, use_cache=True)
        seq.past_kv = out.past_key_values
        next_tok = int(out.logits[0, -1].argmax().item())
        seq.append_token(next_tok)
        seq.first_token_ts = time.time()
        self._emit_token(seq, next_tok)

    def _decode_one(self, seq: Sequence, torch_mod) -> None:
        last_tok = seq.output_ids[-1]
        input_ids = torch_mod.tensor([[last_tok]], device=self._device)
        with torch_mod.inference_mode():
            out = self._model(
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
