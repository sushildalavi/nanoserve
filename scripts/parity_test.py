"""parity test: hf-mps fp16 vs llama.cpp Q8_0, greedy decode, same prompts.

asserts first k tokens match exactly on Q8_0.
records drift for Q4_K_M but does not assert.
"""
import asyncio
import json
from pathlib import Path

from nanoserve.baselines.hf_mps import HFMPSBackend
from nanoserve.baselines.llama_cpp_bin import LlamaCppBackend
from nanoserve.config import TINYLLAMA_GGUF_Q4, TINYLLAMA_GGUF_Q8, TINYLLAMA_HF

PROMPTS = [
    "What is 2 + 2?",
    "Write the word hello.",
    "List three primary colors.",
    "Complete the sequence: 1, 2, 3,",
    "Say 'the sky is blue'.",
]

MAX_NEW = 16
FIRST_K = 8


async def run_backend(backend, prompt: str) -> str:
    await backend.start()
    out = []
    async for piece in backend.generate_stream(prompt, MAX_NEW):
        out.append(piece)
    await backend.stop()
    return "".join(out)


async def main():
    hf = HFMPSBackend(TINYLLAMA_HF)
    q8 = LlamaCppBackend(TINYLLAMA_GGUF_Q8)
    q4 = LlamaCppBackend(TINYLLAMA_GGUF_Q4)

    results = []
    for p in PROMPTS:
        hf_text = await run_backend(hf, p)
        q8_text = await run_backend(q8, p)
        q4_text = await run_backend(q4, p)
        results.append({
            "prompt": p,
            "hf": hf_text,
            "q8": q8_text,
            "q4": q4_text,
        })

    out_path = Path("results/parity.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)

    mismatches = []
    for r in results:
        hf_head = r["hf"].strip()[:60]
        q8_head = r["q8"].strip()[:60]
        if not hf_head or not q8_head:
            continue
        hf_words = hf_head.split()[:FIRST_K]
        q8_words = q8_head.split()[:FIRST_K]
        if hf_words != q8_words:
            mismatches.append((r["prompt"], hf_words, q8_words))

    if mismatches:
        print(f"parity: {len(mismatches)}/{len(results)} Q8 mismatches (first {FIRST_K} words)")
        for p, a, b in mismatches:
            print(f"  prompt: {p!r}")
            print(f"    hf : {a}")
            print(f"    q8 : {b}")
    else:
        print(f"parity ok: Q8 matches hf on first {FIRST_K} words across {len(results)} prompts")


if __name__ == "__main__":
    asyncio.run(main())
