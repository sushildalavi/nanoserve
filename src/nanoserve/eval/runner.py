"""orchestrates the eval sweep: for each quant mode, loads the model,
computes perplexity + hellaswag accuracy, appends a row to
`results/eval.csv`. MPS is preferred; falls back to CPU (which is slow
but works for debugging).
"""
from __future__ import annotations

import csv
import datetime as dt
import gc
import platform
import time
from pathlib import Path

import torch

from nanoserve.config import RESULTS_DIR, TINYLLAMA_HF
from nanoserve.eval.hellaswag import load_items, score_items
from nanoserve.eval.perplexity import compute_perplexity, load_corpus, load_model

EVAL_CSV = RESULTS_DIR / "eval.csv"
EVAL_FIELDS = [
    "timestamp",
    "model",
    "quant_mode",
    "device",
    "ppl",
    "ppl_corpus",
    "ppl_tokens",
    "hs_accuracy",
    "hs_n",
    "hs_source",
    "load_seconds",
    "ppl_seconds",
    "hs_seconds",
    "torch_version",
    "host",
]


def _pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _free_model(model) -> None:
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def run_eval(
    quant_modes: list[str],
    model_path: str = TINYLLAMA_HF.path,
    max_hs_items: int = 100,
    prefer_wikitext: bool = True,
    prefer_hellaswag: bool = True,
    append_csv: bool = True,
) -> list[dict]:
    """run the full eval across `quant_modes`, return a list of per-mode
    result dicts. optionally appends each row to `results/eval.csv`.
    """
    device = _pick_device()
    corpus, ppl_corpus_name = load_corpus(prefer_wikitext=prefer_wikitext)
    hs_items, hs_source = load_items(
        prefer_hellaswag=prefer_hellaswag, max_items=max_hs_items
    )

    if append_csv:
        Path(EVAL_CSV).parent.mkdir(parents=True, exist_ok=True)
        need_header = not EVAL_CSV.exists() or EVAL_CSV.stat().st_size == 0
        csv_fh = open(EVAL_CSV, "a", newline="")
        writer = csv.DictWriter(csv_fh, fieldnames=EVAL_FIELDS)
        if need_header:
            writer.writeheader()
    else:
        csv_fh = None
        writer = None

    rows: list[dict] = []
    try:
        for mode in quant_modes:
            is_mlx = "mlx" in mode  # accepts "mlx-int4", "int4-mlx", etc.
            device_str = "mlx" if is_mlx else str(device)
            print(f"==> eval {mode} on {device_str}")

            t0 = time.perf_counter()
            if is_mlx:
                from nanoserve.eval.mlx_eval import load_mlx_model
                model, tokenizer = load_mlx_model(mode, model_path)
            else:
                model, tokenizer = load_model(mode, model_path, device)
            t_load = time.perf_counter() - t0

            try:
                if is_mlx:
                    from nanoserve.eval.mlx_eval import (
                        compute_perplexity_mlx,
                        score_items_mlx,
                    )
                    t0 = time.perf_counter()
                    ppl = compute_perplexity_mlx(model, tokenizer, corpus)
                    t_ppl = time.perf_counter() - t0

                    t0 = time.perf_counter()
                    hs = score_items_mlx(model, tokenizer, hs_items)
                    t_hs = time.perf_counter() - t0
                else:
                    t0 = time.perf_counter()
                    ppl = compute_perplexity(model, tokenizer, corpus, device)
                    t_ppl = time.perf_counter() - t0

                    t0 = time.perf_counter()
                    hs = score_items(model, tokenizer, hs_items, device)
                    t_hs = time.perf_counter() - t0
            finally:
                # free the loaded model BEFORE the next mode's load_model
                # runs even if compute_perplexity / score_items raises,
                # otherwise a single bad mode can OOM the whole sweep.
                if not is_mlx:
                    _free_model(model)
                else:
                    # mlx uses its own allocator; drop the python ref.
                    del model
                    gc.collect()

            row = {
                "timestamp": dt.datetime.now(dt.UTC)
                    .replace(microsecond=0)
                    .isoformat()
                    .replace("+00:00", "Z"),
                "model": Path(model_path).name,
                "quant_mode": mode,
                "device": device_str,
                "ppl": f"{ppl['ppl']:.4f}",
                "ppl_corpus": ppl_corpus_name,
                "ppl_tokens": ppl["tokens"],
                "hs_accuracy": f"{hs['accuracy']:.4f}",
                "hs_n": int(hs["n"]),
                "hs_source": hs_source,
                "load_seconds": f"{t_load:.2f}",
                "ppl_seconds": f"{t_ppl:.2f}",
                "hs_seconds": f"{t_hs:.2f}",
                "torch_version": torch.__version__,
                "host": platform.node(),
            }
            rows.append(row)
            if writer is not None:
                writer.writerow(row)
                csv_fh.flush()

            print(
                f"    ppl={row['ppl']:>8}  hs_acc={row['hs_accuracy']:>6}  "
                f"({int(hs['n'])} items, {ppl_corpus_name}/{hs_source})"
            )
    finally:
        if csv_fh is not None:
            csv_fh.close()

    return rows
