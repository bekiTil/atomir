"""Episodic write path: extraction -> events -> state-phrased projection, with
write-ahead + idempotent replay."""

from __future__ import annotations

from atomir.embeddings.fake import FakeEmbedder
from atomir.episodic.engine import EpisodicMemory
from atomir.episodic.json_store import JsonEpisodicStore
from atomir.episodic.models import BranchRecord, Event, new_id, now_iso
from atomir.stores.json_store import JsonMemoryStore
from doubles import ScriptedLLM

WORKS_AT_BRANCH = {"branch": "works_at", "state_template": "works at",
                   "description": "employment relationship"}


def _ev(value, polarity="start", verb="works at", modality="happened", occurred_at=None):
    return {"verb_phrase": verb, "value": value, "subject": "the user",
            "subject_type": "person", "object_type": "organization",
            "polarity": polarity, "modality": modality, "occurred_at": occurred_at,
            "raw_text": f"{verb} {value}"}


def _mem(tmp_path, extract_responses, branch_names=None):
    facts = JsonMemoryStore(path=str(tmp_path / "facts.json"))
    episodic = JsonEpisodicStore(path=str(tmp_path / "ep.json"))
    responses = {"extract": extract_responses}
    responses["branch_name"] = branch_names or [WORKS_AT_BRANCH]
    llm = ScriptedLLM(responses=responses)
    mem = EpisodicMemory(facts, episodic, llm, FakeEmbedder(),
                         branch_auto=0.8, branch_gray_low=0.4)
    return mem, facts, episodic, llm


def test_write_creates_episode_events_and_fact(tmp_path):
    mem, facts, episodic, _ = _mem(tmp_path, [{"events": [_ev("Acme Corp")]}])
    res = mem.add("u", "I work at Acme Corp.")
    assert episodic.get_episode("u", res["episode_id"]) is not None
    events = episodic.events("u")
    assert len(events) == 1 and events[0].projected is True and events[0].fact_id
    all_facts = facts.all("u")
    assert len(all_facts) == 1 and all_facts[0]["text"] == "The user works at Acme Corp"


def test_beta_to_acme_leaves_one_live_fact_with_history(tmp_path):
    """Existing Beta employment + new Acme start -> ONE live fact
    (Acme), Beta in history. Never two live employers."""
    mem, facts, _, _ = _mem(tmp_path, [
        {"events": [_ev("Beta Inc")]},
        {"events": [_ev("Acme Corp", polarity="update")]},
    ])
    mem.add("u", "I work at Beta Inc.")
    mem.add("u", "Actually I moved to Acme Corp.")
    all_facts = facts.all("u")
    assert len(all_facts) == 1
    assert all_facts[0]["text"] == "The user works at Acme Corp"
    assert "The user works at Beta Inc" in all_facts[0]["history"]


def test_end_polarity_negates_state_never_deletes(tmp_path):
    """End -> negated-state fact (UPDATE), not a delete."""
    mem, facts, _, _ = _mem(tmp_path, [
        {"events": [_ev("Beta Inc")]},
        {"events": [_ev("Beta Inc", polarity="end")]},
    ])
    mem.add("u", "I work at Beta Inc.")
    mem.add("u", "I left Beta Inc.")
    all_facts = facts.all("u")
    assert len(all_facts) == 1
    assert all_facts[0]["text"] == "The user no longer works at Beta Inc"
    assert "The user works at Beta Inc" in all_facts[0]["history"]


def test_multi_event_message_is_order_independent(tmp_path):
    """'left Beta' + 'joined Acme' in one message -> works at Acme, regardless
    of the order the two events are processed in."""
    for events in (
        [_ev("Beta Inc", polarity="end"), _ev("Acme Corp", polarity="start")],
        [_ev("Acme Corp", polarity="start"), _ev("Beta Inc", polarity="end")],
    ):
        mem, facts, _, _ = _mem(tmp_path, [{"events": events}])
        mem.add("u", "I left Beta and joined Acme.")
        live = facts.all("u")
        assert len(live) == 1, events
        assert live[0]["text"] == "The user works at Acme Corp", events
        # fresh store next iteration
        tmp_path = tmp_path / "next"
        tmp_path.mkdir(exist_ok=True)


def test_planned_modality_stores_event_but_no_fact(tmp_path):
    """R11: planned/hypothetical -> event logged, no fact projected."""
    mem, facts, episodic, _ = _mem(
        tmp_path, [{"events": [_ev("Acme Corp", modality="planned")]}])
    res = mem.add("u", "I might join Acme Corp next year.")
    assert res["facts"] == []
    assert facts.all("u") == []
    events = episodic.events("u")
    assert len(events) == 1 and events[0].modality == "planned"
    assert events[0].projected is True  # marked done, just no fact


def test_replay_is_idempotent(tmp_path):
    """Crash after event append, before projection -> replay projects
    exactly once; a second replay is a no-op."""
    facts = JsonMemoryStore(path=str(tmp_path / "facts.json"))
    episodic = JsonEpisodicStore(path=str(tmp_path / "ep.json"))
    mem = EpisodicMemory(facts, episodic, ScriptedLLM(), FakeEmbedder(),
                         branch_auto=0.8, branch_gray_low=0.4)
    # Simulate a crash: branch + event persisted, but the event never projected.
    episodic.add_branch(BranchRecord(branch="works_at", user_id="u", entity_id="ent_self",
                                     description="emp", state_template="works at"))
    episodic.append_event(Event(id=new_id("ev"), user_id="u", entity_id="ent_self",
                                branch="works_at", value="Acme Corp", raw_text="x",
                                polarity="start", recorded_at=now_iso(), projected=False))
    assert mem.replay("u") == 1
    assert len(facts.all("u")) == 1
    assert mem.replay("u") == 0            # nothing left unprojected
    assert len(facts.all("u")) == 1        # still exactly one fact
