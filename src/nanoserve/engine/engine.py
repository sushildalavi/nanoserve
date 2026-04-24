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
        quant_mode: str = "none",
        admission_policy: str = "fcfs",
        prefix_cache_capacity: int = 0,
    ):
        self.model_spec = model_spec
        # device chosen at start() time so we can fall back to cpu in
        # environments without MPS (linux containers, CI). on Apple
        # Silicon native, MPS is selected and all phase 1-5 numbers
        # remain reproducible.
        self._device: str = "cpu"
        self.quant_mode = quant_mode
        self.weight_bytes_saved: int = 0
        self.num_layers_quantized: int = 0
        # prefix cache. capacity=0 disables (the default — phase 3c flag-flip).
        if prefix_cache_capacity > 0:
            from nanoserve.engine.prefix_cache import PrefixCache
            self.prefix_cache = PrefixCache(capacity=prefix_cache_capacity)
        else:
            self.prefix_cache = None
        self._tokenizer = None
        self._model = None
        self._scheduler = Scheduler(
            cfg=SchedulerConfig(
                max_batch_size=max_batch_size,
                batching_mode=batching_mode,
                admission_policy=admission_policy,
            )
        )
        self._streams: dict[int, asyncio.Queue] = {}
        self._sched_lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._driver: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        # actual forward-pass counters. scheduler stats track "how many seqs
        # were active" each iteration, which is different from "how many
        # forwards actually ran as a single batched call vs N separate
        # single-seq calls". this distinction matters for interpreting
        # continuous-mode results under poisson load.
        self.batched_forward_steps: int = 0
        self.single_forward_steps: int = 0
        # per-step wall-clock timings. forward_ms = time spent inside
        # model.forward() (the matmul-heavy part). step_overhead_ms = time
        # in the same driver iteration outside model.forward (token
        # decoding, queue puts, scheduler bookkeeping). together they let
        # us tell whether the engine is gpu-bound or cpu-bound.
        self.forward_ms: list[float] = []
        self.step_overhead_ms: list[float] = []

    async def start(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # prefer MPS when on Apple Silicon; fall back to CPU otherwise
        # (containerized / CI). cpu serving is dog-slow (1-2 tok/s on
        # TinyLlama) but keeps the serve endpoint functional.
        if torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"

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

        if self.quant_mode == "int8":
            from nanoserve.engine.quant import quantize_model_int8_weight_only
            n, saved = quantize_model_int8_weight_only(self._model)
            self.num_layers_quantized = n
            self.weight_bytes_saved = saved
        elif self.quant_mode == "int4":
            from nanoserve.engine.quant_int4 import quantize_model_int4_weight_only
            n, saved = quantize_model_int4_weight_only(self._model)
            self.num_layers_quantized = n
            self.weight_bytes_saved = saved
        elif self.quant_mode == "torchao_int8":
            # torchao's tensor-subclass int8 weight-only path. on MPS this
            # still goes through fp16 matmul (no native int8 kernel) but
            # uses torchao's optimized dequant path, which may be cheaper
            # than our hand-rolled module's per-call dequant.
            from torchao.quantization import int8_weight_only, quantize_
            quantize_(self._model, int8_weight_only())
            self.num_layers_quantized = -1  # torchao manages this internally
        elif self.quant_mode != "none":
            raise ValueError(f"unknown quant_mode: {self.quant_mode}")
        if self.quant_mode != "none":
            import gc
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

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
        ids = await self.submit_many([req])
        return ids[0]

    async def submit_many(self, reqs: list[SubmitRequest]) -> list[int]:
        """submit multiple requests under one scheduler lock. the driver
        therefore sees all of them in `waiting` when it next runs admit,
        which is what lets fixed-shape batched forwards fire on day 4.
        """
        if self._tokenizer is None:
            raise RuntimeError("engine not started")

        seqs: list[Sequence] = []
        for req in reqs:
            if req.prompt_ids is not None:
                prompt_ids = list(req.prompt_ids)
            else:
                messages = [{"role": "user", "content": req.prompt}]
                templated = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompt_ids = self._tokenizer.encode(
                    templated, add_special_tokens=False
                )
            seqs.append(
                Sequence(
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
            )

        for s in seqs:
            self._streams[s.id] = asyncio.Queue()
        with self._sched_lock:
            for s in seqs:
                self._scheduler.submit(s)
        self._wake.set()
        return [s.id for s in seqs]

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

            # if any admitted seq has a prefix-cache hit, fall back to per-seq
            # prefill — we can't easily mix cached prefixes into a batched
            # forward (different starting cache states). this is fine: the
            # cached portion is skipped entirely so per-seq prefill is fast.
            # we also probe the cache here without consuming the hit (lookup
            # increments hits/misses) — we leave the actual lookup to _prefill.
            any_prefix_hit = False
            if self.prefix_cache is not None:
                for s in admitted:
                    # peek without affecting hit/miss accounting: scan
                    # entries inline rather than calling lookup.
                    for entry in self.prefix_cache._entries.values():
                        cap = min(len(entry.prefix_ids), len(s.prompt_ids) - 1)
                        if cap < self.prefix_cache.MIN_LCP_FOR_HIT:
                            continue
                        match = 0
                        while match < cap and entry.prefix_ids[match] == s.prompt_ids[match]:
                            match += 1
                        if match >= self.prefix_cache.MIN_LCP_FOR_HIT:
                            any_prefix_hit = True
                            break
                    if any_prefix_hit:
                        break

            if (
                len(admitted) > 1
                and self._scheduler.cfg.batching_mode == "continuous"
                and not any_prefix_hit
            ):
                self._prefill_batch(admitted, torch)
                for seq in admitted:
                    seq.admit_ts = time.time()
                    with self._sched_lock:
                        self._scheduler.mark_prefill_done(seq)
            else:
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

            step_t0 = time.time()
            forward_count_before = len(self.forward_ms)

            if self._can_batch_forward(batch):
                self._decode_batch(batch, torch)
                self.batched_forward_steps += 1
            else:
                for seq in batch:
                    self._decode_one(seq, torch)
                    self.single_forward_steps += 1

            for seq in batch:
                stop = seq.should_stop()
                if stop is not None:
                    seq.finish_ts = time.time()
                    with self._sched_lock:
                        self._scheduler.mark_finished(seq, reason=stop)
                    self._emit_done(seq, stop)

            step_total_ms = (time.time() - step_t0) * 1000.0
            forward_total_ms = sum(
                self.forward_ms[forward_count_before:]
            )
            overhead_ms = max(0.0, step_total_ms - forward_total_ms)
            self.step_overhead_ms.append(overhead_ms)

    def _has_pending_work(self) -> bool:
        with self._sched_lock:
            return self._scheduler.has_pending_work()

    def _prefill(self, seq: Sequence, torch_mod) -> None:
        from copy import deepcopy

        from transformers import DynamicCache

        prefix_hit = (
            self.prefix_cache.lookup(seq.prompt_ids)
            if self.prefix_cache is not None
            else None
        )

        if prefix_hit is not None:
            entry, lcp = prefix_hit
            # slice the cached past_kv to lcp tokens (entry may have been
            # stored at a longer length; we only want the matching prefix).
            if lcp == len(entry.prefix_ids):
                past = deepcopy(entry.cached_kv)
            else:
                past = self._slice_cache(entry.cached_kv, lcp, DynamicCache)
            suffix_ids = seq.prompt_ids[lcp:]
            assert suffix_ids, "lookup() must return only when suffix is non-empty"
            input_ids = torch_mod.tensor([suffix_ids], device=self._device)
            with torch_mod.inference_mode():
                out = self._model(
                    input_ids=input_ids,
                    past_key_values=past,
                    use_cache=True,
                )
            seq.past_kv = out.past_key_values
            seq.pad_len = entry.pad_len
        else:
            input_ids = torch_mod.tensor([seq.prompt_ids], device=self._device)
            with torch_mod.inference_mode():
                out = self._model(input_ids=input_ids, use_cache=True)
            seq.past_kv = out.past_key_values
            seq.pad_len = 0
            # store the freshly-prefilled cache for future hits.
            if self.prefix_cache is not None:
                self.prefix_cache.store(seq.prompt_ids, seq.past_kv, pad_len=0)

        next_tok = int(out.logits[0, -1].argmax().item())
        seq.append_token(next_tok)
        seq.first_token_ts = time.time()
        self._emit_token(seq, next_tok)

    @staticmethod
    def _slice_cache(cache, k: int, DynamicCacheCls):
        """return a new DynamicCache containing only the first k tokens of
        each layer's keys/values. used by the prefix cache to truncate a
        longer cached prefix to a shorter LCP match before reuse.
        """
        out = DynamicCacheCls()
        num_layers = len(cache)
        for layer in range(num_layers):
            out.update(
                cache.layers[layer].keys[:, :, :k, :].contiguous(),
                cache.layers[layer].values[:, :, :k, :].contiguous(),
                layer,
            )
        return out

    def _prefill_batch(self, seqs: list[Sequence], torch_mod) -> None:
        """batched prefill with left-padding. each seq ends up with a cache
        of length max_prompt_len; seq.pad_len records how many of those are
        pad tokens so later decode steps can mask them correctly.
        """
        from transformers import DynamicCache

        pad_id = self._tokenizer.pad_token_id
        max_len = max(len(s.prompt_ids) for s in seqs)

        padded_ids = []
        attn_mask = []
        for s in seqs:
            pad = max_len - len(s.prompt_ids)
            padded_ids.append([pad_id] * pad + list(s.prompt_ids))
            attn_mask.append([0] * pad + [1] * len(s.prompt_ids))
            s.pad_len = pad

        input_ids = torch_mod.tensor(padded_ids, device=self._device)
        attention_mask = torch_mod.tensor(attn_mask, device=self._device)

        with torch_mod.inference_mode():
            out = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )

        # split cache per seq. each one carries max_len entries (with pad_len
        # leading entries being from padding positions).
        per_seq_caches = self._split_cache(out.past_key_values, len(seqs), DynamicCache)

        now = time.time()
        for i, s in enumerate(seqs):
            s.past_kv = per_seq_caches[i]
            next_tok = int(out.logits[i, -1].argmax().item())
            s.append_token(next_tok)
            s.first_token_ts = now
            self._emit_token(s, next_tok)

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
        """one forward pass for N sequences with same physical cache length.
        stacks per-seq caches, builds an attention_mask that zeroes out the
        leading pad_len positions of each seq, runs one model.forward,
        splits the updated cache back to each seq.
        """
        from transformers import DynamicCache

        last_tokens = [[s.output_ids[-1]] for s in batch]
        input_ids = torch_mod.tensor(last_tokens, device=self._device)

        batched = self._stack_caches([s.past_kv for s in batch], DynamicCache)
        cache_len = int(batched.get_seq_length())

        # mask: [N, cache_len + 1]. for each row, first pad_len entries are
        # 0, everything after is 1 (real prompt tokens, generated tokens,
        # and the current input token).
        attn_mask = []
        for s in batch:
            row = [0] * s.pad_len + [1] * (cache_len + 1 - s.pad_len)
            attn_mask.append(row)
        attention_mask = torch_mod.tensor(attn_mask, device=self._device)

        fwd_t0 = time.time()
        with torch_mod.inference_mode():
            out = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=batched,
                use_cache=True,
            )
            torch_mod.mps.synchronize()
        self.forward_ms.append((time.time() - fwd_t0) * 1000.0)

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
        transformers DynamicCache exposes per-layer tensors via cache.layers[i].keys / .values.
        """
        import torch

        merged = DynamicCacheCls()
        num_layers = len(caches[0])
        for layer in range(num_layers):
            keys = torch.cat([c.layers[layer].keys for c in caches], dim=0)
            values = torch.cat([c.layers[layer].values for c in caches], dim=0)
            merged.update(keys, values, layer)
        return merged

    @staticmethod
    def _split_cache(batched, n: int, DynamicCacheCls) -> list:
        """split a batched DynamicCache back into n per-seq caches by slicing
        along the batch dim of each layer's keys/values.
        """
        num_layers = len(batched)
        out = []
        for i in range(n):
            per = DynamicCacheCls()
            for layer in range(num_layers):
                per.update(
                    batched.layers[layer].keys[i : i + 1].contiguous(),
                    batched.layers[layer].values[i : i + 1].contiguous(),
                    layer,
                )
            out.append(per)
        return out

    def _decode_one(self, seq: Sequence, torch_mod) -> None:
        last_tok = seq.output_ids[-1]
        input_ids = torch_mod.tensor([[last_tok]], device=self._device)
        kwargs = {"past_key_values": seq.past_kv, "use_cache": True}
        if seq.pad_len > 0:
            cache_len = int(seq.past_kv.get_seq_length())
            mask = [0] * seq.pad_len + [1] * (cache_len + 1 - seq.pad_len)
            kwargs["attention_mask"] = torch_mod.tensor([mask], device=self._device)
        fwd_t0 = time.time()
        with torch_mod.inference_mode():
            out = self._model(input_ids=input_ids, **kwargs)
            torch_mod.mps.synchronize()
        self.forward_ms.append((time.time() - fwd_t0) * 1000.0)
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
