"""SQLite episodic backend. The store contract is run against BOTH
backends so JSON and SQLite stay behaviorally identical."""

from __future__ import annotations

import pytest

from atomir.embeddings.fake import FakeEmbedder
from atomir.episodic.engine import EpisodicMemory
from atomir.episodic.json_store import JsonEpisodicStore
from atomir.episodic.models import (
    BranchRecord, EntityRecord, Episode, Event, new_id, now_iso,
)
from atomir.episodic.sqlite_store import SqliteEpisodicStore
from atomir.stores.json_store import JsonMemoryStore
from doubles import ScriptedLLM


def _make(kind, tmp_path):
    if kind == "json":
        return JsonEpisodicStore(path=str(tmp_path / "ep.json"))
    return SqliteEpisodicStore(path=str(tmp_path / "ep.db"))


@pytest.fixture(params=["json", "sqlite"])
def store(request, tmp_path):
    return _make(request.param, tmp_path)


def _event(user="u", entity="ent_1", branch="works_at", value="Acme",
           polarity="start", occurred_at=None, recorded_at="2026-02-01T00:00:00+00:00", **kw):
    return Event(id=new_id("ev"), user_id=user, entity_id=entity, branch=branch,
                 value=value, raw_text=value, polarity=polarity,
                 recorded_at=recorded_at, occurred_at=occurred_at, **kw)


def test_episode_roundtrip(store):
    ep = Episode(id=new_id("ep"), user_id="u", text="hi", created_at=now_iso())
    store.add_episode(ep)
    assert store.get_episode("u", ep.id).text == "hi"
    assert store.get_episode("other", ep.id) is None
    assert store.delete_episode("u", ep.id) is True
    assert store.get_episode("u", ep.id) is None


def test_events_ordered_and_unprojected(store):
    store.append_event(_event(value="Acme", occurred_at="2025-12-01"))
    store.append_event(_event(value="Beta", occurred_at="2025-11-01"))
    assert [e.value for e in store.events("u")] == ["Beta", "Acme"]
    unp = store.unprojected_events("u")
    assert len(unp) == 2
    e = unp[0]
    e.projected = True
    store.update_event(e)
    assert len(store.unprojected_events("u")) == 1


def test_events_filtered(store):
    store.append_event(_event(entity="ent_1", branch="works_at", value="Acme"))
    store.append_event(_event(entity="ent_1", branch="lives_in", value="Paris"))
    store.append_event(_event(entity="ent_2", branch="works_at", value="Beta"))
    assert {e.value for e in store.events("u", entity_id="ent_1")} == {"Acme", "Paris"}
    assert {e.value for e in store.events("u", branch="works_at")} == {"Acme", "Beta"}


def test_entity_alias_and_branch_ops(store):
    store.add_entity(EntityRecord(entity_id="ent_1", user_id="u",
                                  canonical_name="Dana", aliases=["Dana", "D"]))
    assert store.entity_by_alias("u", "d").entity_id == "ent_1"
    store.add_branch(BranchRecord(branch="works_at", user_id="u", entity_id="ent_1",
                                  description="emp", state_template="works at"))
    assert store.get_branch("u", "ent_1", "works_at").state_template == "works at"
    assert {b.branch for b in store.branches("u", "ent_1")} == {"works_at"}
    assert store.delete_branch("u", "ent_1", "works_at") is True
    assert store.branches("u", "ent_1") == []


def test_clear_is_user_scoped(store):
    store.append_event(_event(user="alice"))
    store.append_event(_event(user="bob"))
    assert store.clear("alice") is True
    assert store.events("alice") == [] and len(store.events("bob")) == 1
    assert store.clear("nobody") is False


@pytest.mark.parametrize("kind", ["json", "sqlite"])
def test_persists_across_reopen(kind, tmp_path):
    s1 = _make(kind, tmp_path)
    s1.append_event(_event(value="Acme"))
    s2 = _make(kind, tmp_path)
    assert [e.value for e in s2.events("u")] == ["Acme"]


def test_engine_end_to_end_on_sqlite(tmp_path):
    facts = JsonMemoryStore(path=str(tmp_path / "facts.json"))
    ep = SqliteEpisodicStore(path=str(tmp_path / "ep.db"))
    llm = ScriptedLLM(responses={
        "extract": [{"events": [{"verb_phrase": "works at", "value": "Acme Corp",
                                 "subject": "the user", "subject_type": "person",
                                 "object_type": "organization", "polarity": "start",
                                 "modality": "happened", "occurred_at": "2025-11-01",
                                 "raw_text": "works at Acme"}]}],
        "branch_name": [{"branch": "works_at", "state_template": "works at",
                         "description": "employment"}]})
    mem = EpisodicMemory(facts, ep, llm, FakeEmbedder(), branch_auto=0.8, branch_gray_low=0.4)
    mem.add("u", "I work at Acme Corp.")
    assert facts.all("u")[0]["text"] == "The user works at Acme Corp"
    assert [e["value"] for e in mem.timeline("u", branch="works_at")] == ["Acme Corp"]
