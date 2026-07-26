"""Episodic read path: typed routing, chain walk, dedup, timeline, and
MemoryService delegation."""

from __future__ import annotations

from atomir.embeddings.fake import FakeEmbedder
from atomir.episodic.engine import EpisodicMemory
from atomir.episodic.json_store import JsonEpisodicStore
from atomir.memory import MemoryService
from atomir.stores.json_store import JsonMemoryStore
from doubles import ScriptedLLM

WORKS_AT = {"branch": "works_at", "state_template": "works at", "description": "employment"}


def _ev(value, polarity="start", occurred_at=None, verb="works at"):
    return {"verb_phrase": verb, "value": value, "subject": "the user",
            "subject_type": "person", "object_type": "organization",
            "polarity": polarity, "modality": "happened", "occurred_at": occurred_at,
            "raw_text": f"{verb} {value}"}


def _build(tmp_path, extract, plans=None, branch_names=None):
    facts = JsonMemoryStore(path=str(tmp_path / "facts.json"))
    episodic = JsonEpisodicStore(path=str(tmp_path / "ep.json"))
    responses = {"extract": extract, "branch_name": branch_names or [WORKS_AT]}
    if plans:
        responses["plan"] = plans
    llm = ScriptedLLM(responses=responses)
    mem = EpisodicMemory(facts, episodic, llm, FakeEmbedder(),
                         branch_auto=0.8, branch_gray_low=0.4)
    return mem, facts, episodic


def test_timeline_ordered_by_occurred_at(tmp_path):
    mem, _, _ = _build(tmp_path, [
        {"events": [_ev("Beta Inc", occurred_at="2025-11-01")]},
        {"events": [_ev("Acme Corp", polarity="update", occurred_at="2025-12-01")]},
    ])
    mem.add("u", "I joined Beta in November.")
    mem.add("u", "I moved to Acme in December.")
    tl = mem.timeline("u")
    assert [e["value"] for e in tl] == ["Beta Inc", "Acme Corp"]
    assert tl[0]["occurred_at"] == "2025-11-01"


def test_timeline_filtered_by_branch(tmp_path):
    mem, _, _ = _build(
        tmp_path,
        [{"events": [_ev("Acme Corp"),
                     {"verb_phrase": "lives in", "value": "Paris", "subject": "the user",
                      "subject_type": "person", "object_type": "city", "polarity": "start",
                      "modality": "happened", "occurred_at": None, "raw_text": "lives in Paris"}]}],
        branch_names=[WORKS_AT,
                      {"branch": "lives_in", "state_template": "lives in", "description": "residence"}],
    )
    mem.add("u", "I work at Acme and live in Paris.")
    assert {e["value"] for e in mem.timeline("u", branch="works_at")} == {"Acme Corp"}
    assert {e["value"] for e in mem.timeline("u", branch="lives_in")} == {"Paris"}


def test_temporal_subquestion_walks_the_chain(tmp_path):
    plan = [{"subquestions": [{"text": "who did the user work for", "type": "temporal",
                              "entity_hint": "the user", "branch_hint": "works_at"}]}]
    mem, _, _ = _build(tmp_path, [
        {"events": [_ev("Beta Inc", occurred_at="2025-11-01")]},
        {"events": [_ev("Acme Corp", polarity="update", occurred_at="2025-12-01")]},
    ], plans=plan)
    mem.add("u", "Beta in November.")
    mem.add("u", "Acme in December.")
    res = mem.search("u", "where has the user worked over time?")
    assert res["subquestion_types"] == ["temporal"]
    vals = [r["value"] for r in res["results"]]
    assert vals == ["Beta Inc", "Acme Corp"]         # chronological chain walk
    assert all(r["type"] == "temporal" for r in res["results"])


def test_current_subquestion_returns_fact(tmp_path):
    plan = [{"subquestions": [{"text": "The user works at Acme Corp", "type": "current",
                              "entity_hint": None, "branch_hint": None}]}]
    mem, _, _ = _build(tmp_path, [{"events": [_ev("Acme Corp")]}], plans=plan)
    mem.add("u", "I work at Acme Corp.")
    res = mem.search("u", "who does the user work for now?")
    assert res["subquestion_types"] == ["current"]
    assert any(r["text"] == "The user works at Acme Corp" for r in res["results"])


def test_event_and_fact_dedup_by_link(tmp_path):
    """A temporal event and the current fact it projected to collapse to one
: the fact isn't shown again after appearing as an event."""
    plan = [{"subquestions": [
        {"text": "when did the user work", "type": "temporal",
         "entity_hint": "the user", "branch_hint": "works_at"},
        {"text": "The user works at Acme Corp", "type": "current",
         "entity_hint": None, "branch_hint": None},
    ]}]
    mem, facts, _ = _build(tmp_path, [{"events": [_ev("Acme Corp")]}], plans=plan)
    mem.add("u", "I work at Acme Corp.")
    fid = facts.all("u")[0]["id"]
    res = mem.search("u", "history and current employer?")
    # The fact id appears once (as the temporal event), not duplicated as current.
    current_hits = [r for r in res["results"] if r.get("type") == "current" and r.get("id") == fid]
    assert current_hits == []
    assert any(r["type"] == "temporal" and r["fact_id"] == fid for r in res["results"])


def test_memoryservice_delegates_to_episodic(tmp_path):
    facts = JsonMemoryStore(path=str(tmp_path / "facts.json"))
    episodic = JsonEpisodicStore(path=str(tmp_path / "ep.json"))
    llm = ScriptedLLM(responses={"extract": [{"events": [_ev("Acme Corp")]}],
                                 "branch_name": [WORKS_AT]})
    engine = EpisodicMemory(facts, episodic, llm, FakeEmbedder(),
                            branch_auto=0.8, branch_gray_low=0.4)
    svc = MemoryService(facts, llm, FakeEmbedder(), episodic=engine)
    svc.add("u", "I work at Acme Corp.")
    assert [e["value"] for e in svc.timeline("u")] == ["Acme Corp"]
    assert svc.reset("u") is True
    assert facts.all("u") == [] and episodic.events("u") == []


def test_timeline_empty_when_episodic_disabled(tmp_path):
    facts = JsonMemoryStore(path=str(tmp_path / "facts.json"))
    svc = MemoryService(facts, ScriptedLLM(), FakeEmbedder())  # no episodic
    assert svc.timeline("u") == []
