import asyncio
import time
from dataclasses import asdict

import psutil
from rich.console import Console

from nanoserve.bench.metrics import RequestRecord, aggregate
from nanoserve.bench.report import append_ablation_row, dump_run, git_commit, now_iso
from nanoserve.bench.workload import build_workload, load_prompts
from nanoserve.config import PROMPTS_DIR, ModelSpec, WorkloadSpec

console = Console()


def _make_backend(model: ModelSpec):
    if model.backend == "hf_mps":
        from nanoserve.baselines.hf_mps import HFMPSBackend
        return HFMPSBackend(model)
    if model.backend == "llama_cpp":
        from nanoserve.baselines.llama_cpp_bin import LlamaCppBackend
        return LlamaCppBackend(model)
    raise ValueError(f"unknown backend: {model.backend}")


async def _drive_one(backend, req, t0: float, results: list[RequestRecord]):
    arrival_wall = t0 + req.arrival_offset_s
    now = time.time()
    if arrival_wall > now:
        await asyncio.sleep(arrival_wall - now)

    arrival_ts = time.time()
    first_ts = 0.0
    start_ts = arrival_ts
    out_text_chunks: list[str] = []
    ok = True
    err = None
    try:
        n_out = 0
        async for tok in backend.generate_stream(req.prompt, req.max_new_tokens):
            if first_ts == 0.0:
                first_ts = time.time()
            out_text_chunks.append(tok)
            n_out += 1
        end_ts = time.time()
        if first_ts == 0.0:
            first_ts = end_ts
    except Exception as e:
        end_ts = time.time()
        if first_ts == 0.0:
            first_ts = end_ts
        ok = False
        err = str(e)
        n_out = 0

    in_toks = backend.count_tokens(req.prompt)
    out_text = "".join(out_text_chunks)
    out_toks = backend.count_tokens(out_text) if out_text else n_out

    results.append(
        RequestRecord(
            idx=req.idx,
            arrival_ts=arrival_ts,
            start_ts=start_ts,
            first_token_ts=first_ts,
            end_ts=end_ts,
            input_tokens=in_toks,
            output_tokens=out_toks,
            ok=ok,
            error=err,
        )
    )


async def _run_async(
    backend, workload: WorkloadSpec, model: ModelSpec
) -> tuple[list[RequestRecord], float, float]:
    prompts = load_prompts(PROMPTS_DIR / workload.prompt_file)
    reqs = build_workload(
        prompts,
        kind=workload.kind,
        num_requests=workload.num_requests,
        rate=workload.rate,
        max_new_tokens=workload.max_new_tokens,
        seed=workload.seed,
    )

    await backend.start()
    proc = psutil.Process()
    mem_peak = proc.memory_info().rss / (1024 * 1024)

    results: list[RequestRecord] = []
    t0 = time.time()

    if workload.kind == "closed-loop":
        sem = asyncio.Semaphore(workload.concurrency)

        async def worker(req):
            async with sem:
                await _drive_one(backend, req, t0, results)

        await asyncio.gather(*(worker(r) for r in reqs))
    elif workload.kind == "single":
        for r in reqs:
            await _drive_one(backend, r, t0, results)
    else:
        tasks = [asyncio.create_task(_drive_one(backend, r, t0, results)) for r in reqs]
        for fut in asyncio.as_completed(tasks):
            await fut
            cur = proc.memory_info().rss / (1024 * 1024)
            if cur > mem_peak:
                mem_peak = cur

    wall = time.time() - t0
    await backend.stop()
    return results, wall, mem_peak


def run_baseline(model: ModelSpec, workload: WorkloadSpec) -> dict:
    backend = _make_backend(model)
    console.print(f"[bold]baseline[/bold] backend={model.backend} model={model.name} quant={model.quant} workload={workload.kind}")
    records, wall, mem_peak = asyncio.run(_run_async(backend, workload, model))
    agg = aggregate(records, wall_s=wall)

    config = {"model": asdict(model), "workload": asdict(workload)}
    tag = f"{model.backend}_{model.quant}_{workload.kind}"
    dump_run(tag=tag, config=config, records=records, agg=agg, mem_peak_mb=mem_peak)

    row = {
        "ts": now_iso(),
        "commit": git_commit(),
        "backend": model.backend,
        "model": model.name,
        "quant": model.quant,
        "batching": "off",
        "paging": "off",
        "prefix_cache": "off",
        "workload": workload.kind,
        "rate": workload.rate,
        "concurrency": workload.concurrency,
        "n": agg.n_ok,
        "rps": round(agg.rps, 3),
        "p50_ttft_ms": round(agg.ttft_p50, 2),
        "p95_ttft_ms": round(agg.ttft_p95, 2),
        "p99_ttft_ms": round(agg.ttft_p99, 2),
        "p50_tpot_ms": round(agg.tpot_p50, 2),
        "p95_tpot_ms": round(agg.tpot_p95, 2),
        "p50_e2e_ms": round(agg.e2e_p50, 2),
        "p95_e2e_ms": round(agg.e2e_p95, 2),
        "p99_e2e_ms": round(agg.e2e_p99, 2),
        "decode_tok_s": round(agg.decode_tok_s, 2),
        "mem_peak_mb": round(mem_peak, 1),
    }
    append_ablation_row(row)

    console.print(row)
    return row
