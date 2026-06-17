from __future__ import annotations

import argparse
import asyncio
import csv
import json
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT = "Write a one-sentence summary of local Apple Silicon inference."


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _format_float(value: str | None, digits: int = 2) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except ValueError:
        return value


def build_matrix(rows: list[dict[str, str]], live_row: dict[str, str] | None = None) -> list[dict[str, str]]:
    matrix: list[dict[str, str]] = []
    for row in rows:
        matrix.append(
            {
                "backend": row.get("backend", ""),
                "quant": row.get("quant", ""),
                "concurrency": row.get("concurrency", ""),
                "metric": "tokens/sec",
                "value": _format_float(row.get("decode_tok_s"), 2),
                "ttft_p95_ms": _format_float(row.get("p95_ttft_ms"), 2),
                "tpot_p95_ms": _format_float(row.get("p95_tpot_ms"), 2),
                "e2e_p95_ms": _format_float(row.get("p95_e2e_ms"), 2),
                "errors": _format_float(row.get("http_req_failed"), 0),
            }
        )
    if live_row:
        matrix.append(live_row)
    return matrix


def render_markdown(matrix: list[dict[str, str]]) -> str:
    lines = [
        "# nanoserve Benchmark Matrix",
        "",
        "| backend | quant | concurrency | tokens/sec | p95 TTFT ms | p95 TPOT ms | p95 E2E ms | errors |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in matrix:
        lines.append(
            "| {backend} | {quant} | {concurrency} | {value} | {ttft_p95_ms} | {tpot_p95_ms} | {e2e_p95_ms} | {errors} |".format(
                **row
            )
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a benchmark matrix from nanoserve CSV artifacts.")
    parser.add_argument("--ablations", default=str(ROOT / "results" / "ablations.csv"))
    parser.add_argument("--output", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--from-existing-results", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--live-url", default="")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


async def _measure_one(client: httpx.AsyncClient, url: str, prompt: str) -> dict[str, float | int]:
    started = asyncio.get_running_loop().time()
    token_count = 0
    first_token_at: float | None = None
    async with client.stream(
        "POST",
        f"{url.rstrip('/')}/v1/chat/completions",
        json={
            "model": "tinyllama-1.1b-chat",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 16,
            "stream": True,
        },
        timeout=60.0,
    ) as response:
        if response.status_code != 200:
            raise RuntimeError(f"live benchmark failed with HTTP {response.status_code}")

        async for line in response.aiter_lines():
            if not line or not line.startswith("data: "):
                continue
            if line.strip() == "data: [DONE]":
                break
            payload = json.loads(line[len("data: ") :])
            delta = payload.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content")
            if content:
                token_count += 1
                if first_token_at is None:
                    first_token_at = asyncio.get_running_loop().time()

    ended = asyncio.get_running_loop().time()
    if first_token_at is None:
        first_token_at = ended
    ttft_ms = (first_token_at - started) * 1000.0
    e2e_ms = (ended - started) * 1000.0
    tpot_ms = (ended - first_token_at) * 1000.0 / max(token_count - 1, 1)
    tok_per_sec = token_count / max(ended - started, 1e-9)
    return {
        "tokens_per_sec": tok_per_sec,
        "ttft_p95_ms": ttft_ms,
        "tpot_p95_ms": tpot_ms,
        "e2e_p95_ms": e2e_ms,
        "errors": 0,
    }


async def _measure_live_row(live_url: str, prompt: str, concurrency: int) -> dict[str, str]:
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *[_measure_one(client, live_url, prompt) for _ in range(max(1, concurrency))],
            return_exceptions=True,
        )

    successes = [r for r in results if isinstance(r, dict)]
    failures = sum(1 for r in results if not isinstance(r, dict))
    if not successes:
        return {
            "backend": live_url,
            "quant": "live",
            "concurrency": str(concurrency),
            "metric": "tokens/sec",
            "value": "",
            "ttft_p95_ms": "",
            "tpot_p95_ms": "",
            "e2e_p95_ms": "",
            "errors": str(failures),
        }

    def _p95(values: list[float]) -> float:
        ordered = sorted(values)
        index = int(round(0.95 * (len(ordered) - 1)))
        return ordered[index]

    return {
        "backend": live_url,
        "quant": "live",
        "concurrency": str(concurrency),
        "metric": "tokens/sec",
        "value": f"{sum(item['tokens_per_sec'] for item in successes) / len(successes):.2f}",
        "ttft_p95_ms": f"{_p95([item['ttft_p95_ms'] for item in successes]):.2f}",
        "tpot_p95_ms": f"{_p95([item['tpot_p95_ms'] for item in successes]):.2f}",
        "e2e_p95_ms": f"{_p95([item['e2e_p95_ms'] for item in successes]):.2f}",
        "errors": str(failures),
    }


def main() -> None:
    args = build_parser().parse_args()
    ablations_path = Path(args.ablations)
    if not ablations_path.is_absolute():
        ablations_path = ROOT / ablations_path

    live_row: dict[str, str] | None = None
    notes: list[str] = []
    if args.live_url and not args.no_download and not args.dry_run:
        try:
            live_row = asyncio.run(_measure_live_row(args.live_url, args.prompt, args.concurrency))
        except Exception as exc:
            notes.append(f"live benchmark unavailable: {exc}")
            live_row = {
                "backend": args.live_url,
                "quant": "live",
                "concurrency": str(args.concurrency),
                "metric": "tokens/sec",
                "value": "",
                "ttft_p95_ms": "",
                "tpot_p95_ms": "",
                "e2e_p95_ms": "",
                "errors": "1",
            }
    elif args.live_url and args.no_download:
        notes.append("live benchmark skipped because --no-download was set")

    rows = _read_rows(ablations_path) if args.from_existing_results else []
    matrix = build_matrix(rows, live_row=live_row)
    markdown = render_markdown(matrix)
    if notes:
        markdown += "\n\n## Notes\n\n" + "\n".join(f"- {note}" for note in notes)

    if args.dry_run:
        print(f"would read {ablations_path}")
        if args.live_url:
            print(f"would benchmark live server at {args.live_url}")
        if args.output:
            print(f"would write markdown to {args.output}")
        if args.output_json:
            print(f"would write json to {args.output_json}")
        return

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = ROOT / output_path
        output_path.write_text(markdown + "\n", encoding="utf-8")
    else:
        print(markdown)

    if args.output_json:
        output_json = Path(args.output_json)
        if not output_json.is_absolute():
            output_json = ROOT / output_json
        output_json.write_text(json.dumps({"matrix": matrix, "notes": notes}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
