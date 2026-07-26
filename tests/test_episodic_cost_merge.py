"""Per-add cost telemetry and the merge_entities primitive."""

from __future__ import annotations

from atomir.embeddings.fake import FakeEmbedder
from atomir.episodic.engine import EpisodicMemory
from atomir.episodic.json_store import JsonEpisodicStore
from atomir.episodic.models import BranchRecord, EntityRecord, Event, new_id, now_iso
from atomir.llm.fake import FakeLLM
from atomir.stores.json_store import JsonMemoryStore
from doubles import ScriptedLLM

WORKS_AT = {"branch": "works_at", "state_template": "works at", "description": "employment"}


def _ev(value, polarity="start"):
    return {"verb_phrase": "works at", "value": value, "subject": "the user",
            "subject_type": "person", "object_type": "organization",
            "polarity": polarity, "modality": "happened", "occurred_at": None,
            "raw_text": f"works at {value}"}


def _mem(tmp_path, llm=None):
    facts = JsonMemoryStore(path=str(tmp_path / "facts.json"))
    episodic = JsonEpisodicStore(path=str(tmp_path / "ep.json"))
    mem = EpisodicMemory(facts, episodic, llm or FakeLLM(), FakeEmbedder(),
                         branch_auto=0.8, branch_gray_low=0.4)
    return mem, facts, episodic


# --- cost instrumentation -------------------------------------------

def test_add_reports_cost_diagnostics(tmp_path):
    llm = ScriptedLLM(responses={"extract": [{"events": [_ev("Acme Corp")]}],
                                 "branch_name": [WORKS_AT]})
    mem, _, _ = _mem(tmp_path, llm)
    diag = mem.add("u", "I work at Acme Corp.")["diagnostics"]
    assert diag["extraction_calls"] == 1          # single extraction
    assert diag["judge_calls"] == 1               # naming the new branch
    assert diag["embedding_calls"] == 1           # one projected fact embedded
    assert diag["total_llm_tokens"] > 0


def test_extraction_stays_one_call_for_multi_event_message(tmp_path):
    llm = ScriptedLLM(responses={
        "extract": [{"events": [_ev("Beta Inc"), _ev("Acme Corp", "update")]}],
        "branch_name": [WORKS_AT]})
    mem, _, _ = _mem(tmp_path, llm)
    diag = mem.add("u", "I moved from Beta to Acme.")["diagnostics"]
    assert diag["extraction_calls"] == 1          # NOT one per event


# --- merge_entities -------------------------------------------------

def _seed_entity(store, entity_id, alias):
    store.add_entity(EntityRecord(entity_id=entity_id, user_id="u",
                                  canonical_name=alias, aliases=[alias]))


def test_merge_reparents_events_branches_and_aliases(tmp_path):
    mem, _, ep = _mem(tmp_path)
    _seed_entity(ep, "ent_dana", "Dana")
    _seed_entity(ep, "ent_d", "D")
    ep.add_branch(BranchRecord(branch="reports_to", user_id="u", entity_id="ent_d",
                               description="mgmt", state_template="reports to"))
    ep.append_event(Event(id=new_id("ev"), user_id="u", entity_id="ent_d",
                          branch="reports_to", value="x", raw_text="x",
                          polarity="start", recorded_at=now_iso()))

    assert mem.merge_entities("u", keep_id="ent_dana", absorb_id="ent_d") is True
    assert ep.get_entity("u", "ent_d") is None                       # absorbed gone
    assert "D" in ep.get_entity("u", "ent_dana").aliases             # alias unioned
    assert len(ep.events("u", entity_id="ent_dana")) == 1            # event re-parented
    assert ep.events("u", entity_id="ent_d") == []
    assert {b.branch for b in ep.branches("u", "ent_dana")} == {"reports_to"}
    assert ep.branches("u", "ent_d") == []


def test_merge_same_named_branch_interleaves_events(tmp_path):
    mem, _, ep = _mem(tmp_path)
    _seed_entity(ep, "ent_keep", "Dana")
    _seed_entity(ep, "ent_abs", "D")
    for eid in ("ent_keep", "ent_abs"):
        ep.add_branch(BranchRecord(branch="works_at", user_id="u", entity_id=eid,
                                   description="emp", state_template="works at"))
    ep.append_event(Event(id=new_id("ev"), user_id="u", entity_id="ent_keep",
                          branch="works_at", value="Dec", raw_text="x", polarity="start",
                          recorded_at=now_iso(), occurred_at="2025-12-01"))
    ep.append_event(Event(id=new_id("ev"), user_id="u", entity_id="ent_abs",
                          branch="works_at", value="Nov", raw_text="x", polarity="start",
                          recorded_at=now_iso(), occurred_at="2025-11-01"))

    assert mem.merge_entities("u", "ent_keep", "ent_abs") is True
    merged = ep.events("u", entity_id="ent_keep", branch="works_at")
    assert [e.value for e in merged] == ["Nov", "Dec"]     # interleaved by occurred_at
    assert len(ep.branches("u", "ent_keep")) == 1          # one works_at branch


def test_merge_missing_entity_is_noop(tmp_path):
    mem, _, ep = _mem(tmp_path)
    _seed_entity(ep, "ent_a", "A")
    assert mem.merge_entities("u", "ent_a", "ent_missing") is False
    assert mem.merge_entities("u", "ent_a", "ent_a") is False  # self-merge refused
