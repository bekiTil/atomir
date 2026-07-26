"""Branch checkpoints."""

from __future__ import annotations

from atomir.embeddings.fake import FakeEmbedder
from atomir.episodic.engine import EpisodicMemory
from atomir.episodic.json_store import JsonEpisodicStore
from atomir.stores.json_store import JsonMemoryStore
from doubles import ScriptedLLM

WORKS_AT = {"branch": "works_at", "state_template": "works at", "description": "employment"}


def _ev(value, occurred_at):
    return {"verb_phrase": "works at", "value": value, "subject": "the user",
            "subject_type": "person", "object_type": "organization", "polarity": "update",
            "modality": "happened", "occurred_at": occurred_at, "raw_text": f"works at {value}"}


def test_checkpoint_summarizes_oldest_segment_keeping_raw_events(tmp_path):
    facts = JsonMemoryStore(path=str(tmp_path / "facts.json"))
    ep = JsonEpisodicStore(path=str(tmp_path / "ep.json"))
    llm = ScriptedLLM(responses={
        "extract": [{"events": [_ev(f"Co{i}", f"2025-01-0{i}")]} for i in range(1, 5)],
        "branch_name": [WORKS_AT]})
    mem = EpisodicMemory(facts, ep, llm, FakeEmbedder(), branch_auto=0.8,
                         branch_gray_low=0.4, checkpoint_every=3)

    for i in range(1, 5):
        mem.add("u", f"Now at Co{i}.")

    all_events = ep.events("u", branch="works_at")
    raw = [e for e in all_events if e.kind == "event"]
    checkpoints = [e for e in all_events if e.kind == "checkpoint"]
    checkpointed = [e for e in raw if e.checkpointed]

    assert len(raw) == 4                         # raw events KEPT, not deleted
    assert len(checkpoints) == 1                 # one summary node created
    assert len(checkpointed) == 3                # oldest segment folded in
    assert llm.calls["summary"] == 1

    # Chain walk = checkpoint summary + the live tail (the checkpointed raw
    # events are hidden behind the summary).
    walk = mem.timeline("u", branch="works_at")
    kinds = [e["kind"] for e in walk]
    assert kinds == ["checkpoint", "event"]
    assert walk[-1]["value"] == "Co4"
    assert walk[0]["text"].startswith("(summary)")
