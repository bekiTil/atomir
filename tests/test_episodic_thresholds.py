"""Per-embedder threshold table + config wiring."""

from __future__ import annotations

import os

from atomir.config import Settings
from atomir.episodic.thresholds import get_thresholds


def test_get_thresholds_per_embedder():
    o = get_thresholds("openai")
    assert set(o) == {"auto", "gray_low", "floor", "margin"}
    assert o["auto"] == 0.72
    assert get_thresholds("jina")["auto"] == 0.80
    # Unknown embedder -> safe fallback.
    assert get_thresholds("nope")["auto"] == 0.80


def test_config_reads_threshold_table(monkeypatch):
    monkeypatch.setenv("EMBED_BACKEND", "openai")
    for k in ("BRANCH_MATCH_AUTO", "BRANCH_RESOLVE_FLOOR", "BRANCH_RESOLVE_MARGIN"):
        monkeypatch.delenv(k, raising=False)
    s = Settings()
    assert s.branch_match_auto == 0.72
    assert s.branch_resolve_floor == 0.30
    assert s.branch_resolve_margin == 0.10


def test_env_override_wins(monkeypatch):
    monkeypatch.setenv("EMBED_BACKEND", "openai")
    monkeypatch.setenv("BRANCH_MATCH_AUTO", "0.66")
    assert Settings().branch_match_auto == 0.66
