"""Cascade deletes / forget."""

from __future__ import annotations

from atomir.embeddings.fake import FakeEmbedder
from atomir.episodic.engine import EpisodicMemory
from atomir.episodic.json_store import JsonEpisodicStore
from atomir.memory import MemoryService
from atomir.stores.json_store import JsonMemoryStore
from doubles import ScriptedLLM

WORKS_AT = {"branch": "works_at", "state_template": "works at", "description": "employment"}
DATED = {"branch": "dated", "state_template": "dated", "description": "relationship"}


def _ev(value, polarity="start", verb="works at", obj="organization"):
    return {"verb_phrase": verb, "value": value, "subject": "the user",
            "subject_type": "person", "object_type": obj, "polarity": polarity,
            "modality": "happened", "occurred_at": None, "raw_text": f"{verb} {value}"}


def _mem(tmp_path, extract, branch_names):
    facts = JsonMemoryStore(path=str(tmp_path / "facts.json"))
    episodic = JsonEpisodicStore(path=str(tmp_path / "ep.json"))
    llm = ScriptedLLM(responses={"extract": extract, "branch_name": branch_names})
    mem = EpisodicMemory(facts, episodic, llm, FakeEmbedder(),
                         branch_auto=0.8, branch_gray_low=0.4)
    return mem, facts, episodic


def test_delete_fact_cascades_to_events_and_episode(tmp_path):
    """Direction (a): deleting a fact removes its source events and orphaned
    episode, and the content is unreachable via get_all / timeline."""
    mem, facts, ep = _mem(tmp_path, [{"events": [_ev("Acme Corp")]}], [WORKS_AT])
    mem.add("u", "I work at Acme Corp.")
    fid = facts.all("u")[0]["id"]

    assert mem.delete_fact("u", fid) is True
    assert facts.all("u") == []                       # fact gone
    assert ep.events("u") == []                       # source event gone
    assert ep._db["episodes"] == []                   # orphaned episode redacted
    assert mem.timeline("u") == []                    # unreachable via timeline
    b = ep.get_branch("u", ep.branches("u")[0].entity_id, "works_at") if ep.branches("u") else None
    assert b is None or b.current_fact_id is None      # branch pointer cleared


def test_delete_event_rederives_fact_from_survivors(tmp_path):
    """Direction (b): deleting the latest event re-derives the branch fact from
    what remains."""
    mem, facts, ep = _mem(tmp_path,
                          [{"events": [_ev("Beta Inc"), _ev("Acme Corp", "update")]}],
                          [WORKS_AT])
    mem.add("u", "Beta then Acme.")
    assert facts.all("u")[0]["text"] == "The user works at Acme Corp"

    acme_event = [e for e in ep.events("u") if e.value == "Acme Corp"][0]
    assert mem.delete_event("u", acme_event.id) is True

    live = facts.all("u")
    assert len(live) == 1
    assert live[0]["text"] == "The user works at Beta Inc"   # re-derived from survivor
    assert [e.value for e in ep.events("u")] == ["Beta Inc"]


def test_forget_entity_removes_all_traces(tmp_path):
    mem, facts, ep = _mem(tmp_path,
                          [{"events": [_ev("Alex", verb="dated", obj="person")]}],
                          [DATED])
    mem.add("u", "I dated Alex.")
    alex = ep.entity_by_alias("u", "Alex")
    assert alex is not None and facts.all("u")

    assert mem.forget_entity("u", alex.entity_id) is True
    assert ep.entity_by_alias("u", "Alex") is None
    assert all(e.value != "Alex" for e in ep.events("u"))
    assert all("Alex" not in f["text"] for f in facts.all("u"))


def test_forget_redacts_name_from_surviving_episode(tmp_path):
    """HARD-GATE piece: an episode that mentions the forgotten entity AND holds
    other events survives, but the name is scrubbed from its raw text."""
    mem, facts, ep = _mem(tmp_path,
                          [{"events": [_ev("Acme Corp"),
                                       _ev("Alex", verb="dated", obj="person")]}],
                          [WORKS_AT, DATED])
    # One message, two events: employment (kept) + Alex (to forget).
    mem.add("u", "I joined Acme Corp and I dated Alex.")
    alex = ep.entity_by_alias("u", "Alex")
    assert mem.forget_entity("u", alex.entity_id) is True

    # Episode survives (still has the Acme event) but 'Alex' is gone from raw text.
    eps = ep.episodes("u")
    assert len(eps) == 1
    assert "Alex" not in eps[0].text and "[forgotten]" in eps[0].text
    assert "Acme" in eps[0].text                       # unrelated content preserved
    # Employment event/fact untouched.
    assert any("Acme" in f["text"] for f in facts.all("u"))


def test_forget_redacts_name_even_from_episode_with_no_event(tmp_path):
    """The instruction case: 'forget X' produces no event about X, so that episode
    isn't 'touched' — redaction must still scrub the name from it."""
    facts = JsonMemoryStore(path=str(tmp_path / "facts.json"))
    ep = JsonEpisodicStore(path=str(tmp_path / "ep.json"))
    llm = ScriptedLLM(responses={
        "extract": [{"events": [_ev("Alex", verb="dated", obj="person")]},
                    {"events": []}],   # 2nd message mentions Alex but yields no event
        "branch_name": [DATED]})
    mem = EpisodicMemory(facts, ep, llm, FakeEmbedder(), branch_auto=0.8, branch_gray_low=0.4)
    mem.add("u", "I dated Alex.")
    mem.add("u", "Please forget everything about Alex, and note I joined Acme.")
    alex = ep.entity_by_alias("u", "Alex")
    assert mem.forget_entity("u", alex.entity_id) is True
    assert all("Alex" not in e.text for e in ep.episodes("u"))   # scrubbed everywhere


def test_memoryservice_forget_by_name(tmp_path):
    facts = JsonMemoryStore(path=str(tmp_path / "facts.json"))
    ep = JsonEpisodicStore(path=str(tmp_path / "ep.json"))
    llm = ScriptedLLM(responses={"extract": [{"events": [_ev("Alex", verb="dated", obj="person")]}],
                                 "branch_name": [DATED]})
    engine = EpisodicMemory(facts, ep, llm, FakeEmbedder(), branch_auto=0.8, branch_gray_low=0.4)
    svc = MemoryService(facts, llm, FakeEmbedder(), episodic=engine)
    svc.add("u", "I dated Alex.")
    assert svc.forget("u", "Alex") is True
    assert svc.forget("u", "Nobody") is False
    assert facts.all("u") == [] and ep.entity_by_alias("u", "Alex") is None
