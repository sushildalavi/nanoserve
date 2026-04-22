import typer

app = typer.Typer(help="nanoserve cli", no_args_is_help=True)
baseline_app = typer.Typer(help="run a baseline backend end-to-end")
bench_app = typer.Typer(help="run benchmark workloads")
app.add_typer(baseline_app, name="baseline")
app.add_typer(bench_app, name="bench")


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
    workload: str = "closed-loop",
    num_requests: int = 20,
    concurrency: int = 1,
    rate: float = 2.0,
    max_new_tokens: int = 128,
):
    """run the nanoserve engine as a backend. flag-flipped ablation: pass
    batching_mode=serial vs continuous, quant_mode=none vs int8, etc.
    """
    from nanoserve.bench.runner import run_baseline
    from nanoserve.config import WorkloadSpec, tinyllama_nanoserve

    run_baseline(
        tinyllama_nanoserve(
            batching_mode=batching_mode,
            max_batch_size=max_batch_size,
            quant_mode=quant_mode,
        ),
        WorkloadSpec(
            kind=workload,
            num_requests=num_requests,
            concurrency=concurrency,
            rate=rate,
            max_new_tokens=max_new_tokens,
        ),
    )


@bench_app.command("sweep")
def bench_sweep():
    typer.echo("sweep not implemented yet")


if __name__ == "__main__":
    app()
