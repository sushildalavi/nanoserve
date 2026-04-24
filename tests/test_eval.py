"""unit tests for the parts of the eval harness that don't need a GPU
or a real LM loaded. keeps CI fast — the model-touching paths are
exercised by `make eval` on the user's Mac.
"""
from __future__ import annotations

from nanoserve.eval.hellaswag import FIXTURE_ITEMS, HSItem, load_items
from nanoserve.eval.perplexity import FIXTURE_PATH, load_corpus


def test_ppl_fixture_exists_and_is_nontrivial():
    assert FIXTURE_PATH.exists()
    text = FIXTURE_PATH.read_text(encoding="utf-8")
    assert len(text) > 2000, f"fixture too small: {len(text)} bytes"


def test_load_corpus_offline_returns_fixture_text():
    text = load_corpus(prefer_wikitext=False)
    assert isinstance(text, str)
    assert len(text) > 2000
    assert "transistor" in text or "computer" in text


def test_load_items_offline_returns_fixture():
    items = load_items(prefer_hellaswag=False)
    assert len(items) == len(FIXTURE_ITEMS)
    for it in items:
        assert isinstance(it, HSItem)
        assert isinstance(it.ctx, str) and len(it.ctx) > 0
        assert len(it.endings) == 4
        assert 0 <= it.label < 4


def test_fixture_items_have_unique_correct_endings():
    """sanity: the 'correct' ending should not be literally identical to
    a wrong one, which would make scoring ambiguous.
    """
    for d in FIXTURE_ITEMS:
        endings = d["endings"]
        assert len(set(endings)) == 4, f"duplicate endings in: {d['ctx']!r}"


def test_fixture_labels_in_range():
    for d in FIXTURE_ITEMS:
        assert 0 <= int(d["label"]) < 4
