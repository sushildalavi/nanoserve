"""cli smoke tests: run every subcommand with --help via Typer's
CliRunner. catches the class of bug where a cli command was renamed
or imported module is missing. no network, no GPU, no model load.
"""
from __future__ import annotations

from typer.testing import CliRunner

from nanoserve.cli import app

runner = CliRunner()


def test_top_level_help():
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0
    # each subapp registered in cli.py should appear in the help output
    for cmd in ("serve", "baseline", "bench", "eval"):
        assert cmd in r.output, f"{cmd} missing from top-level help: {r.output}"


def test_baseline_help_has_three_backends():
    r = runner.invoke(app, ["baseline", "--help"])
    assert r.exit_code == 0
    for backend in ("hf", "llamacpp", "nanoserve"):
        assert backend in r.output


def test_eval_help_has_both_commands():
    r = runner.invoke(app, ["eval", "--help"])
    assert r.exit_code == 0
    assert "all" in r.output
    assert "ppl" in r.output


def test_eval_all_help_advertises_supported_quant_modes():
    r = runner.invoke(app, ["eval", "all", "--help"])
    assert r.exit_code == 0
    # the docstring/help lists the supported quant modes
    for mode in ("fp16", "int8", "int4"):
        assert mode in r.output


def test_serve_help_lists_flag_knobs():
    r = runner.invoke(app, ["serve", "--help"])
    assert r.exit_code == 0
    for flag in ("batching-mode", "max-batch-size", "quant-mode", "admission-policy"):
        assert flag in r.output
