from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"
PROMPTS_DIR = REPO_ROOT / "prompts"


@dataclass
class ModelSpec:
    name: str
    backend: str
    path: str
    quant: str = "fp16"
    # nanoserve-engine only
    batching_mode: str = "serial"
    max_batch_size: int = 1
    quant_mode: str = "none"  # "none" | "int8" — nanoserve engine internal quant


@dataclass
class WorkloadSpec:
    kind: str
    num_requests: int = 100
    concurrency: int = 4
    rate: float = 2.0
    max_new_tokens: int = 128
    seed: int = 0
    prompt_file: str = "bench_prompts.jsonl"


@dataclass
class RunConfig:
    model: ModelSpec
    workload: WorkloadSpec
    tags: dict = field(default_factory=dict)


TINYLLAMA_HF = ModelSpec(
    name="tinyllama-1.1b-chat",
    backend="hf_mps",
    path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    quant="fp16",
)

TINYLLAMA_GGUF_Q8 = ModelSpec(
    name="tinyllama-1.1b-chat",
    backend="llama_cpp",
    path="models/tinyllama-1.1b-chat.Q8_0.gguf",
    quant="Q8_0",
)

TINYLLAMA_GGUF_Q4 = ModelSpec(
    name="tinyllama-1.1b-chat",
    backend="llama_cpp",
    path="models/tinyllama-1.1b-chat.Q4_K_M.gguf",
    quant="Q4_K_M",
)


def tinyllama_nanoserve(
    batching_mode: str,
    max_batch_size: int,
    quant_mode: str = "none",
) -> ModelSpec:
    return ModelSpec(
        name="tinyllama-1.1b-chat",
        backend="nanoserve",
        path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        quant="fp16" if quant_mode == "none" else quant_mode,
        batching_mode=batching_mode,
        max_batch_size=max_batch_size,
        quant_mode=quant_mode,
    )
