"""Integration test over a multi-event message.

Deferred to later phases (asserted there, not here): forget-cascade (§8.3,
), branch checkpoints (§8.7), and entity v2 "D" -> Dana merge
(§8.8 — v1 under-merges by design, covered by test_episodic_registry).
"""

from __future__ import annotations

from atomir.embeddings.fake import FakeEmbedder
from atomir.episodic.engine import EpisodicMemory
from atomir.episodic.json_store import JsonEpisodicStore
from atomir.stores.json_store import JsonMemoryStore
from doubles import ScriptedLLM

MESSAGE = ("Please forget everything about my ex Alex. Anyway, big news I forgot to "
          "mention — I actually left Beta back in November and joined Acme Corp, so D "
          "isn't my manager anymore since she stayed at Beta. Oh and I've kept up my "
          "daily runs.")

# What a good extractor returns for the message (one call). Employment uses one
# predicate with start/end polarity so both land on the SAME works_at branch.
EVENTS = {"events": [
    {"verb_phrase": "works at", "value": "Beta Inc", "subject": "the user",
     "subject_type": "person", "object_type": "organization", "polarity": "end",
     "modality": "happened", "occurred_at": "2025-11-01", "raw_text": "left Beta in November"},
    {"verb_phrase": "works at", "value": "Acme Corp", "subject": "the user",
     "subject_type": "person", "object_type": "organization", "polarity": "start",
     "modality": "happened", "occurred_at": "2025-11-01", "raw_text": "joined Acme Corp"},
    {"verb_phrase": "reports to", "value": "Dana", "subject": "the user",
     "subject_type": "person", "object_type": "person", "polarity": "end",
     "modality": "happened", "occurred_at": None, "raw_text": "D isn't my manager anymore"},
    {"verb_phrase": "does", "value": "daily runs", "subject": "the user",
     "subject_type": "person", "object_type": "activity", "polarity": "start",
     "modality": "happened", "occurred_at": None, "raw_text": "kept up my daily runs"},
]}
BRANCH_NAMES = [
    {"branch": "works_at", "state_template": "works at", "description": "employment"},
    {"branch": "reports_to", "state_template": "reports to", "description": "management"},
    {"branch": "runs", "state_template": "does", "description": "activity"},
]


def _mem(tmp_path):
    facts = JsonMemoryStore(path=str(tmp_path / "facts.json"))
    episodic = JsonEpisodicStore(path=str(tmp_path / "ep.json"))
    llm = ScriptedLLM(responses={"extract": [EVENTS], "branch_name": list(BRANCH_NAMES)})
    mem = EpisodicMemory(facts, episodic, llm, FakeEmbedder(),
                         branch_auto=0.8, branch_gray_low=0.4)
    return mem, facts, episodic, llm


def test_multi_event_message(tmp_path):
    mem, facts, episodic, llm = _mem(tmp_path)
    res = mem.add("u", MESSAGE)

    # — exactly ONE extraction call for the whole message.
    assert llm.calls["extract"] == 1

    self_entity = episodic.entity_by_alias("u", "the user")
    assert self_entity is not None
    branch_names = {b.branch for b in episodic.branches("u", self_entity.entity_id)}

    # — 'left Beta' and 'joined Acme' land on ONE works_at branch; no spurious
    # employment branch. Both events present with end/start polarity.
    assert "works_at" in branch_names
    emp = episodic.events("u", branch="works_at")
    assert [e.value for e in emp] == ["Beta Inc", "Acme Corp"]
    assert {e.polarity for e in emp} == {"start", "end"}

    # Beta->Acme leaves Acme as the single live employer; Beta is not a live fact.
    fact_texts = [f["text"] for f in facts.all("u")]
    assert "The user works at Acme Corp" in fact_texts
    assert all("Beta" not in t for t in fact_texts)
    # — identical spelling across event and fact.
    assert emp[1].value == "Acme Corp" and "Acme Corp" in "The user works at Acme Corp"

    # — event time is November (not the message date), timeline ordered.
    assert all(e.occurred_at == "2025-11-01" for e in emp)
    tl = mem.timeline("u", branch="works_at")
    assert [e["value"] for e in tl] == ["Beta Inc", "Acme Corp"]

    # Daily-run activity is captured as its own branch + fact.
    assert "runs" in branch_names
    assert "The user does daily runs" in fact_texts

    # — replay is a consistent no-op once everything is projected.
    before = sorted(fact_texts)
    assert mem.replay("u") == 0
    assert sorted(f["text"] for f in facts.all("u")) == before


def test_replay_after_simulated_crash(tmp_path):
    """§8.2: crash between event append and projection -> replay reconciles both
    layers with no duplication."""
    mem, facts, episodic, _ = _mem(tmp_path)
    mem.add("u", MESSAGE)
    # Simulate a crash that lost projection on the employment events.
    for e in episodic.events("u", branch="works_at"):
        e.projected, e.fact_id = False, None
        episodic.update_event(e)
    live_before = {f["text"] for f in facts.all("u")}
    mem.replay("u")
    # No duplicate employment facts; Acme still the single live employer.
    emp_facts = [f["text"] for f in facts.all("u") if "works at" in f["text"] or "work" in f["text"]]
    assert emp_facts.count("The user works at Acme Corp") <= 1
    assert live_before == {f["text"] for f in facts.all("u")}
