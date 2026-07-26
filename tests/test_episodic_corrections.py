"""Corrections: `corrects` supersedes the mistaken event, no double-count
."""

from __future__ import annotations

from atomir.embeddings.fake import FakeEmbedder
from atomir.episodic.engine import EpisodicMemory
from atomir.episodic.json_store import JsonEpisodicStore
from atomir.stores.json_store import JsonMemoryStore
from doubles import ScriptedLLM

WORKS_AT = {"branch": "works_at", "state_template": "works at", "description": "employment"}


def _ev(value, occurred_at=None, corrects_hint=None):
    return {"verb_phrase": "works at", "value": value, "subject": "the user",
            "subject_type": "person", "object_type": "organization", "polarity": "start",
            "modality": "happened", "occurred_at": occurred_at,
            "corrects_hint": corrects_hint, "raw_text": f"works at {value}"}


def test_correction_supersedes_and_does_not_double_count(tmp_path):
    facts = JsonMemoryStore(path=str(tmp_path / "facts.json"))
    ep = JsonEpisodicStore(path=str(tmp_path / "ep.json"))
    llm = ScriptedLLM(responses={
        "extract": [
            {"events": [_ev("Acme Corp", occurred_at="2025-10-01")]},
            {"events": [_ev("Acme Corp", occurred_at="2025-11-01",
                            corrects_hint="actually joined Acme in November")]},
        ],
        "branch_name": [WORKS_AT]})
    mem = EpisodicMemory(facts, ep, llm, FakeEmbedder(), branch_auto=0.8, branch_gray_low=0.4)

    mem.add("u", "I joined Acme in October.")
    mem.add("u", "Correction — I actually joined Acme in November.")

    events = ep.events("u", branch="works_at")
    superseded = [e for e in events if e.superseded]
    live = [e for e in events if not e.superseded]
    assert len(superseded) == 1 and superseded[0].occurred_at == "2025-10-01"
    assert len(live) == 1 and live[0].occurred_at == "2025-11-01"
    assert live[0].corrects == superseded[0].id

    # Timeline shows only the corrected truth; the fact isn't double-counted.
    tl = mem.timeline("u", branch="works_at")
    assert [e["occurred_at"] for e in tl] == ["2025-11-01"]
    assert len(facts.all("u")) == 1
    assert facts.all("u")[0]["text"] == "The user works at Acme Corp"
