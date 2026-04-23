import typer

app = typer.Typer(help="nanoserve cli", no_args_is_help=True)
baseline_app = typer.Typer(help="run a baseline backend end-to-end")
bench_app = typer.Typer(help="run benchmark workloads")
app.add_typer(baseline_app, name="baseline")
app.add_typer(bench_app, name="bench")


@app.command("serve")
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    batching_mode: str = "continuous",
    max_batch_size: int = 4,
    quant_mode: str = "none",
    admission_policy: str = "synchronized",
    prefix_cache_capacity: int = 16,
):
    """start the openai-compatible api server. flags are passed to the
    engine via env vars so the same defaults work whether you set them
    here or via NANOSERVE_* in the deployment environment.
    """
    import os

    import uvicorn

    os.environ["NANOSERVE_BATCHING_MODE"] = batching_mode
    os.environ["NANOSERVE_MAX_BATCH_SIZE"] = str(max_batch_size)
    os.environ["NANOSERVE_QUANT_MODE"] = quant_mode
    os.environ["NANOSERVE_ADMISSION_POLICY"] = admission_policy
    os.environ["NANOSERVE_PREFIX_CACHE_CAPACITY"] = str(prefix_cache_capacity)

    typer.echo(
        f"starting nanoserve api on {host}:{port} "
        f"(batching={batching_mode}, max_batch={max_batch_size}, "
        f"quant={quant_mode}, admit={admission_policy}, "
        f"prefix_cache={prefix_cache_capacity})"
    )
    uvicorn.run("nanoserve.server.api:app", host=host, port=port, log_level="info")


@baseline_app.command("hf")
def baseline_hf(
    workload: str = "poisson",
    num_requests: int = 100,
    concurrency: int = 4,
    rate: float = 2.0,
    max_new_tokens: int = 128,
):
    from nanoserve.bench.runner import run_baseline
    from nanoserve.config import TINYLLAMA_HF, WorkloadSpec

    run_baseline(
        TINYLLAMA_HF,
        WorkloadSpec(
            kind=workload,
            num_requests=num_requests,
            concurrency=concurrency,
            rate=rate,
            max_new_tokens=max_new_tokens,
        ),
    )


@baseline_app.command("llamacpp")
def baseline_llamacpp(
    quant: str = "Q8_0",
    workload: str = "poisson",
    num_requests: int = 100,
    concurrency: int = 4,
    rate: float = 2.0,
    max_new_tokens: int = 128,
):
    from nanoserve.bench.runner import run_baseline
    from nanoserve.config import TINYLLAMA_GGUF_Q4, TINYLLAMA_GGUF_Q8, WorkloadSpec

    model = TINYLLAMA_GGUF_Q8 if quant == "Q8_0" else TINYLLAMA_GGUF_Q4
    run_baseline(
        model,
        WorkloadSpec(
            kind=workload,
            num_requests=num_requests,
            concurrency=concurrency,
            rate=rate,
            max_new_tokens=max_new_tokens,
        ),
    )


@baseline_app.command("nanoserve")
def baseline_nanoserve(
    batching_mode: str = "serial",
    max_batch_size: int = 1,
    quant_mode: str = "none",
    admission_policy: str = "fcfs",
    prefix_cache_capacity: int = 0,
    workload: str = "closed-loop",
    num_requests: int = 20,
    concurrency: int = 1,
    rate: float = 2.0,
    max_new_tokens: int = 128,
    shared_prefix_tokens: int = 0,
):
    """run the nanoserve engine as a backend. flag-flipped ablation: pass
    batching_mode=serial|continuous, quant_mode=none|int8,
    admission_policy=fcfs|synchronized, etc.
    """
    from nanoserve.bench.runner import run_baseline
    from nanoserve.config import WorkloadSpec, tinyllama_nanoserve

    run_baseline(
        tinyllama_nanoserve(
            batching_mode=batching_mode,
            max_batch_size=max_batch_size,
            quant_mode=quant_mode,
            admission_policy=admission_policy,
            prefix_cache_capacity=prefix_cache_capacity,
        ),
        WorkloadSpec(
            kind=workload,
            num_requests=num_requests,
            concurrency=concurrency,
            rate=rate,
            max_new_tokens=max_new_tokens,
            shared_prefix_tokens=shared_prefix_tokens,
        ),
    )


@bench_app.command("sweep")
def bench_sweep():
    typer.echo("sweep not implemented yet")


if __name__ == "__main__":
    app()
