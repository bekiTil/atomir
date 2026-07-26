"""Unit tests for the episodic models + JSON store."""

from __future__ import annotations

from atomir.episodic.json_store import JsonEpisodicStore
from atomir.episodic.models import (
    BranchRecord,
    EntityRecord,
    Episode,
    Event,
    new_id,
    now_iso,
)


def _store(tmp_path):
    return JsonEpisodicStore(path=str(tmp_path / "ep.json"))


def _event(user="u", entity="ent_1", branch="works_at", value="Acme",
           polarity="start", occurred_at=None, recorded_at="2026-02-01T00:00:00+00:00",
           **kw):
    return Event(id=new_id("ev"), user_id=user, entity_id=entity, branch=branch,
                 value=value, raw_text=value, polarity=polarity,
                 recorded_at=recorded_at, occurred_at=occurred_at, **kw)


def test_episode_roundtrip_and_delete(tmp_path):
    s = _store(tmp_path)
    ep = Episode(id=new_id("ep"), user_id="u", text="hello world", created_at=now_iso())
    s.add_episode(ep)
    got = s.get_episode("u", ep.id)
    assert got is not None and got.text == "hello world"
    assert s.get_episode("other", ep.id) is None  # user-scoped
    assert s.delete_episode("u", ep.id) is True
    assert s.get_episode("u", ep.id) is None


def test_events_ordered_by_occurred_then_recorded(tmp_path):
    s = _store(tmp_path)
    # recorded in Feb, but occurred in Nov -> Nov must sort before a Dec event.
    nov = _event(value="Beta", occurred_at="2025-11-01T00:00:00+00:00")
    dec = _event(value="Acme", occurred_at="2025-12-01T00:00:00+00:00")
    s.append_event(dec)
    s.append_event(nov)
    ordered = [e.value for e in s.events("u")]
    assert ordered == ["Beta", "Acme"]  # by occurred_at, not append order


def test_events_fallback_to_recorded_when_occurred_none(tmp_path):
    s = _store(tmp_path)
    a = _event(value="A", occurred_at=None, recorded_at="2026-01-01T00:00:00+00:00")
    b = _event(value="B", occurred_at=None, recorded_at="2026-01-02T00:00:00+00:00")
    s.append_event(b)
    s.append_event(a)
    assert [e.value for e in s.events("u")] == ["A", "B"]


def test_events_filter_by_entity_and_branch(tmp_path):
    s = _store(tmp_path)
    s.append_event(_event(entity="ent_1", branch="works_at", value="Acme"))
    s.append_event(_event(entity="ent_1", branch="lives_in", value="Paris"))
    s.append_event(_event(entity="ent_2", branch="works_at", value="Beta"))
    assert {e.value for e in s.events("u", entity_id="ent_1")} == {"Acme", "Paris"}
    assert {e.value for e in s.events("u", branch="works_at")} == {"Acme", "Beta"}
    assert {e.value for e in s.events("u", entity_id="ent_1", branch="works_at")} == {"Acme"}


def test_unprojected_and_update_projected_flag(tmp_path):
    s = _store(tmp_path)
    e = _event()
    s.append_event(e)
    assert [x.id for x in s.unprojected_events("u")] == [e.id]
    e.projected = True
    s.update_event(e)
    assert s.unprojected_events("u") == []
    assert s.get_event("u", e.id).projected is True


def test_entity_alias_casefold_lookup(tmp_path):
    s = _store(tmp_path)
    s.add_entity(EntityRecord(entity_id="ent_1", user_id="u",
                              canonical_name="Dana", aliases=["Dana", "my manager"]))
    assert s.entity_by_alias("u", "dana").entity_id == "ent_1"
    assert s.entity_by_alias("u", "  MY MANAGER ").entity_id == "ent_1"
    assert s.entity_by_alias("u", "Sam") is None
    assert s.entity_by_alias("other", "Dana") is None  # user-scoped


def test_entity_update_adds_alias(tmp_path):
    s = _store(tmp_path)
    e = EntityRecord(entity_id="ent_1", user_id="u", canonical_name="Dana", aliases=["Dana"])
    s.add_entity(e)
    e.aliases.append("D")
    s.update_entity(e)
    assert s.entity_by_alias("u", "D").entity_id == "ent_1"


def test_branch_registry_scoped_per_entity(tmp_path):
    s = _store(tmp_path)
    s.add_branch(BranchRecord(branch="works_at", user_id="u", entity_id="ent_1",
                              description="employment", state_template="works at",
                              aliases=["joined", "left"]))
    s.add_branch(BranchRecord(branch="lives_in", user_id="u", entity_id="ent_1",
                              description="residence", state_template="lives in"))
    got = s.get_branch("u", "ent_1", "works_at")
    assert got is not None and got.state_template == "works at"
    assert {b.branch for b in s.branches("u", entity_id="ent_1")} == {"works_at", "lives_in"}
    assert s.get_branch("u", "ent_2", "works_at") is None


def test_clear_is_user_scoped(tmp_path):
    s = _store(tmp_path)
    s.append_event(_event(user="alice"))
    s.add_episode(Episode(id=new_id("ep"), user_id="alice", text="x", created_at=now_iso()))
    s.append_event(_event(user="bob"))
    assert s.clear("alice") is True
    assert s.events("alice") == [] and s.get_episode("alice", "any") is None
    assert len(s.events("bob")) == 1
    assert s.clear("nobody") is False


def test_persists_across_reopen(tmp_path):
    path = str(tmp_path / "ep.json")
    s1 = JsonEpisodicStore(path=path)
    s1.append_event(_event(value="Acme"))
    s2 = JsonEpisodicStore(path=path)
    assert [e.value for e in s2.events("u")] == ["Acme"]
