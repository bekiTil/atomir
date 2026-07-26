"""Value-anchored end resolution + repair_branches."""

from __future__ import annotations

from atomir.embeddings.fake import FakeEmbedder
from atomir.episodic.engine import EpisodicMemory
from atomir.episodic.json_store import JsonEpisodicStore
from atomir.episodic.models import BranchRecord, Event, new_id, now_iso
from atomir.stores.json_store import JsonMemoryStore
from doubles import ScriptedLLM


def _ev(verb, value, polarity):
    return {"verb_phrase": verb, "value": value, "subject": "the user",
            "subject_type": "person", "object_type": "organization", "polarity": polarity,
            "modality": "happened", "occurred_at": None, "raw_text": f"{verb} {value}"}


def _mem(tmp_path, extract, branch_names):
    facts = JsonMemoryStore(path=str(tmp_path / "facts.json"))
    ep = JsonEpisodicStore(path=str(tmp_path / "ep.json"))
    llm = ScriptedLLM(responses={"extract": extract, "branch_name": branch_names})
    mem = EpisodicMemory(facts, ep, llm, FakeEmbedder(), branch_auto=0.8, branch_gray_low=0.4)
    return mem, facts, ep, llm


def test_end_event_anchors_to_start_branch_despite_different_verb(tmp_path):
    """'left Beta' resolves to the branch that STARTED Beta, even though the
    namer would give it a different canonical — no second branch is created."""
    mem, _, ep, llm = _mem(
        tmp_path,
        [{"events": [_ev("joined", "Beta Inc", "start")]},
         {"events": [_ev("left", "Beta", "end")]}],
        # only ONE namer response: if value-anchoring failed, the 2nd add would
        # need another and fork a branch -> the test would see 2 branches.
        [{"branch": "joined_org", "state_template": "works at", "description": "employment"}])
    mem.add("u", "I joined Beta Inc.")
    mem.add("u", "I left Beta.")

    self_e = ep.entity_by_alias("u", "the user")
    branches = ep.branches("u", self_e.entity_id)
    assert len(branches) == 1                       # ONE employment branch
    evs = ep.events("u", branch=branches[0].branch)
    assert {e.polarity for e in evs} == {"start", "end"}
    assert [e.value for e in evs] == ["Beta Inc", "Beta"]
    assert llm.calls["branch_name"] == 1            # namer fired only for the start


def test_value_anchored_alias_is_quarantined(tmp_path):
    """The verb of a value-anchored end goes to provisional_aliases, NOT the
    confirmed set — so 'left' can't later exact-match employment on its own."""
    mem, _, ep, _ = _mem(
        tmp_path,
        [{"events": [_ev("joined", "Beta Inc", "start")]},
         {"events": [_ev("left", "Beta", "end")]}],
        [{"branch": "joined_org", "state_template": "works at", "description": "employment"}])
    mem.add("u", "I joined Beta Inc.")
    mem.add("u", "I left Beta.")
    self_e = ep.entity_by_alias("u", "the user")
    b = ep.branches("u", self_e.entity_id)[0]
    assert "left" in b.provisional_aliases        # quarantined
    assert "left" not in b.aliases                # NOT confirmed

    # A fresh 'left' with an unrelated value must NOT exact-match this branch via
    # the provisional alias.
    from atomir.episodic.registry import BranchMatcher
    from atomir.embeddings.fake import FakeEmbedder
    m = BranchMatcher(ep, mem.llm, FakeEmbedder(), auto=0.8, gray_low=0.4)
    res = m.match("u", self_e.entity_id, "left", "person", "activity", {"the gym"},
                  value="the gym", polarity="end")
    assert res["zone"] != "exact"                 # provisional alias didn't fire


def test_template_sanitizer_strips_object_type_word(tmp_path):
    from atomir.episodic.registry import _strip_type_words
    assert _strip_type_words("joined organization") == "joined"
    assert _strip_type_words("lives in place") == "lives in"
    assert _strip_type_words("works at") == "works at"


def test_end_without_prior_start_takes_normal_path(tmp_path):
    mem, _, ep, _ = _mem(
        tmp_path, [{"events": [_ev("left", "Acme", "end")]}],
        [{"branch": "left_acme", "state_template": "left", "description": "x"}])
    mem.add("u", "I left Acme.")     # no prior start -> normal path, must not crash
    self_e = ep.entity_by_alias("u", "the user")
    assert len(ep.branches("u", self_e.entity_id)) == 1


def test_repair_branches_heals_a_pre_split_store(tmp_path):
    facts = JsonMemoryStore(path=str(tmp_path / "facts.json"))
    ep = JsonEpisodicStore(path=str(tmp_path / "ep.json"))
    mem = EpisodicMemory(facts, ep, ScriptedLLM(), FakeEmbedder(),
                         branch_auto=0.8, branch_gray_low=0.4)
    # A store already split: works_at started Beta; a stray 'left_b' holds the end.
    ep.add_branch(BranchRecord(branch="works_at", user_id="u", entity_id="ent_self",
                               description="emp", state_template="works at",
                               object_type="organization"))
    ep.append_event(Event(id=new_id("ev"), user_id="u", entity_id="ent_self",
                          branch="works_at", value="Beta Inc", raw_text="x",
                          polarity="start", recorded_at=now_iso(), projected=True))
    ep.add_branch(BranchRecord(branch="left_b", user_id="u", entity_id="ent_self",
                               description="left", state_template="left",
                               object_type="organization"))
    ep.append_event(Event(id=new_id("ev"), user_id="u", entity_id="ent_self",
                          branch="left_b", value="Beta", raw_text="x",
                          polarity="end", recorded_at=now_iso(), projected=True))

    assert mem.repair_branches("u") == 1
    assert {b.branch for b in ep.branches("u", "ent_self")} == {"works_at"}
    evs = ep.events("u", branch="works_at")
    assert {e.polarity for e in evs} == {"start", "end"}
