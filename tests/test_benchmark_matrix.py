from __future__ import annotations

from scripts.benchmark_matrix import build_matrix, render_markdown


def test_build_matrix_parses_expected_fields():
    matrix = build_matrix(
        [
            {
                "backend": "mlx",
                "quant": "int4",
                "concurrency": "1",
                "decode_tok_s": "136.26",
                "p95_ttft_ms": "69.94",
                "p95_tpot_ms": "7.84",
                "p95_e2e_ms": "563.28",
                "http_req_failed": "0",
            }
        ]
    )
    assert matrix[0]["backend"] == "mlx"
    assert matrix[0]["value"] == "136.26"
    assert matrix[0]["ttft_p95_ms"] == "69.94"


def test_render_markdown_includes_headers():
    markdown = render_markdown(
        [
            {
                "backend": "mlx",
                "quant": "int4",
                "concurrency": "1",
                "value": "136.26",
                "ttft_p95_ms": "69.94",
                "tpot_p95_ms": "7.84",
                "e2e_p95_ms": "563.28",
                "errors": "0",
            }
        ]
    )
    assert "nanoserve Benchmark Matrix" in markdown
    assert "136.26" in markdown


def test_build_matrix_appends_live_row():
    matrix = build_matrix(
        [
            {
                "backend": "mlx",
                "quant": "int4",
                "concurrency": "1",
                "decode_tok_s": "136.26",
                "p95_ttft_ms": "69.94",
                "p95_tpot_ms": "7.84",
                "p95_e2e_ms": "563.28",
                "http_req_failed": "0",
            }
        ],
        live_row={
            "backend": "http://127.0.0.1:8000",
            "quant": "live",
            "concurrency": "1",
            "metric": "tokens/sec",
            "value": "11.00",
            "ttft_p95_ms": "20.00",
            "tpot_p95_ms": "2.00",
            "e2e_p95_ms": "200.00",
            "errors": "0",
        },
    )
    assert matrix[-1]["backend"].startswith("http")
    assert matrix[-1]["value"] == "11.00"
