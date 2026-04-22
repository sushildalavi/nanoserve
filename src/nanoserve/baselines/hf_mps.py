import asyncio
from collections.abc import AsyncIterator
from threading import Thread

from nanoserve.baselines.base import Backend
from nanoserve.config import ModelSpec


class HFMPSBackend(Backend):
    """naive serving: one request at a time through hf .generate with a streamer.
    this is the intentional 'before' baseline — no batching, no paging.
    """

    name = "hf_mps"

    def __init__(self, model: ModelSpec):
        self.model_spec = model
        self._lock = asyncio.Lock()
        self._tokenizer = None
        self._model = None
        self._device = "mps"

    async def start(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.backends.mps.is_available():
            raise RuntimeError("mps not available on this machine")

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_spec.path)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_spec.path,
            dtype=torch.float16,
            attn_implementation="eager",
        ).to(self._device)
        self._model.eval()

    async def stop(self) -> None:
        self._model = None
        self._tokenizer = None

    def count_tokens(self, text: str) -> int:
        if self._tokenizer is None or not text:
            return 0
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    async def generate_stream(self, prompt: str, max_new_tokens: int) -> AsyncIterator[str]:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("backend not started")

        from transformers import TextIteratorStreamer

        tok = self._tokenizer
        model = self._model

        messages = [{"role": "user", "content": prompt}]
        input_text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tok(input_text, return_tensors="pt").to(self._device)

        streamer = TextIteratorStreamer(
            tok, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            streamer=streamer,
            pad_token_id=tok.pad_token_id,
        )

        async with self._lock:
            loop = asyncio.get_running_loop()
            thread = Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)
            thread.start()

            q: asyncio.Queue = asyncio.Queue()
            sentinel = object()

            def drain():
                try:
                    for piece in streamer:
                        loop.call_soon_threadsafe(q.put_nowait, piece)
                finally:
                    loop.call_soon_threadsafe(q.put_nowait, sentinel)

            Thread(target=drain, daemon=True).start()

            while True:
                item = await q.get()
                if item is sentinel:
                    break
                yield item
            thread.join()
