"""NanoServeEngine — one engine, two modes (serial / continuous).

architecture: one persistent background thread drives the scheduler. the
event loop submits requests and consumes per-seq token queues. this lets
multiple callers stream concurrently.

day 4 scope: when batching_mode is 'continuous' AND every currently
decoding seq shares the same past_kv length, a single batched
model.forward runs one step for all of them. otherwise the driver falls
back to per-seq decode. fixed-shape 2A constraint: same prompt length,
same max_new_tokens, no mid-batch retirement — all active seqs stay in
sync. mixed-length lands in 2B with masks.
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

            # step every currently-decoding sequence.
            with self._sched_lock:
                batch = self._scheduler.pick_decode_batch()
            if not batch:
                continue

            if self._can_batch_forward(batch):
                self._decode_batch(batch, torch)
            else:
                for seq in batch:
                    self._decode_one(seq, torch)

            for seq in batch:
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

    def _can_batch_forward(self, batch: list[Sequence]) -> bool:
        """batched forward is safe only when the scheduler is in continuous
        mode AND every seq's cache has identical length. this is the fixed-
        shape 2A constraint. mixed lengths require masks (2B).
        """
        if self._scheduler.cfg.batching_mode != "continuous" or len(batch) < 2:
            return False
        first_len = self._cache_len(batch[0])
        return all(self._cache_len(s) == first_len for s in batch[1:])

    @staticmethod
    def _cache_len(seq: Sequence) -> int:
        cache = seq.past_kv
        if cache is None:
            return 0
        try:
            return int(cache.get_seq_length())
        except AttributeError:
            # legacy tuple-of-tuples cache
            return cache[0][0].shape[-2]

    def _decode_batch(self, batch: list[Sequence], torch_mod) -> None:
        """one forward pass for N same-length sequences.
        stacks per-seq caches along the batch dim, runs model.forward once,
        then splits the updated cache back onto each seq.
        """
        from transformers import DynamicCache

        last_tokens = [[s.output_ids[-1]] for s in batch]
        input_ids = torch_mod.tensor(last_tokens, device=self._device)

        batched = self._stack_caches([s.past_kv for s in batch], DynamicCache)

        with torch_mod.inference_mode():
            out = self._model(
                input_ids=input_ids,
                past_key_values=batched,
                use_cache=True,
            )

        new_caches = self._split_cache(out.past_key_values, len(batch), DynamicCache)

        logits = out.logits  # [N, 1, vocab]
        next_tokens = logits[:, -1].argmax(dim=-1).tolist()

        for i, seq in enumerate(batch):
            seq.past_kv = new_caches[i]
            seq.append_token(int(next_tokens[i]))
            self._emit_token(seq, int(next_tokens[i]))

    @staticmethod
    def _stack_caches(caches: list, DynamicCacheCls):
        """concat per-seq DynamicCache objects along batch dim. all inputs
        must share the same seq_len per layer (2A precondition).
        """
        import torch

        merged = DynamicCacheCls()
        num_layers = len(caches[0].key_cache)
        for layer in range(num_layers):
            keys = torch.cat([c.key_cache[layer] for c in caches], dim=0)
            values = torch.cat([c.value_cache[layer] for c in caches], dim=0)
            merged.update(keys, values, layer)
        return merged

    @staticmethod
    def _split_cache(batched, n: int, DynamicCacheCls) -> list:
        """split a batched DynamicCache back into n per-seq caches."""
        out = []
        num_layers = len(batched.key_cache)
        for i in range(n):
            per = DynamicCacheCls()
            for layer in range(num_layers):
                per.update(
                    batched.key_cache[layer][i : i + 1],
                    batched.value_cache[layer][i : i + 1],
                    layer,
                )
            out.append(per)
        return out

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
