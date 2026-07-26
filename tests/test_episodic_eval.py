"""Smoke test for the temporal efficiency eval: episodic routing must
beat flat hybrid on temporal questions (higher recall, fewer tokens) without
hurting current questions."""

from __future__ import annotations

from eval.episodic.temporal_eval import run_eval
from eval.episodic.distractor_eval import generate_corpus, run_config, QUESTIONS


def test_episodic_wins_temporal_efficiency():
    rows = run_eval()
    temporal = [r for r in rows if r["type"] == "temporal"]
    current = [r for r in rows if r["type"] == "current"]
    assert temporal and current

    for r in temporal:
        assert r["epi_recall"] >= r["base_recall"]     # never worse recall
        assert r["epi_tokens"] < r["base_tokens"]      # always cheaper context

    # At least one temporal question the baseline outright misses (former state
    # only exists in the event log).
    assert any(r["base_recall"] < 1.0 <= r["epi_recall"] for r in temporal)

    # Current-state questions: episodic doesn't regress recall.
    for r in current:
        assert r["epi_recall"] >= r["base_recall"]

    epi = sum(r["epi_recall"] for r in temporal) / len(temporal)
    base = sum(r["base_recall"] for r in temporal) / len(temporal)
    assert epi > base


def test_distractor_harness_runs_and_scales():
    """Plumbing smoke for the distractor eval (fake providers): corpus generates,
    metrics compute, and the corpus is actually large (the whole point)."""
    from atomir.embeddings.fake import FakeEmbedder
    from atomir.llm.fake import FakeLLM

    msgs = generate_corpus(60, seed=7)
    assert len(msgs) == 60
    assert len(QUESTIONS) >= 20
    s = run_config(FakeLLM(), FakeEmbedder(), "OFF", msgs, k=6)
    for typ in ("temporal", "current", "semantic"):
        assert typ in s and 0.0 <= s[typ]["recall"] <= 1.0
    assert "_mechanisms" in s and "_ingest_s" in s
