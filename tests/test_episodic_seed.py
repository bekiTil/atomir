"""Seed ontology."""

from __future__ import annotations

from atomir.embeddings.fake import FakeEmbedder
from atomir.episodic.engine import EpisodicMemory
from atomir.episodic.json_store import JsonEpisodicStore
from atomir.ontology import apply_pack, normalize_object_type
from atomir.ontology.personal import PERSONAL_SEEDS
from atomir.stores.json_store import JsonMemoryStore
from doubles import ScriptedLLM


def _ev(verb, value, obj, polarity="start"):
    return {"verb_phrase": verb, "value": value, "subject": "the user",
            "subject_type": "person", "object_type": obj, "polarity": polarity,
            "modality": "happened", "occurred_at": None, "raw_text": f"{verb} {value}"}


def _seeded_mem(tmp_path, extract, branch_names=None):
    facts = JsonMemoryStore(path=str(tmp_path / "facts.json"))
    ep = JsonEpisodicStore(path=str(tmp_path / "ep.json"))
    llm = ScriptedLLM(responses={"extract": extract, "branch_name": branch_names or []})
    mem = EpisodicMemory(facts, ep, llm, FakeEmbedder(), branch_auto=0.8,
                         branch_gray_low=0.4, ontology_pack="personal")
    return mem, facts, ep, llm


def test_normalize_object_type_maps_to_enum():
    assert normalize_object_type("org") == "organization"
    assert normalize_object_type("city") == "place"
    assert normalize_object_type("vehicle") == "object"
    assert normalize_object_type("wombat") == "other"
    assert normalize_object_type("") == "other"


def test_apply_pack_is_idempotent(tmp_path):
    ep = JsonEpisodicStore(path=str(tmp_path / "ep.json"))
    assert apply_pack(ep, "u", "ent_self", PERSONAL_SEEDS) == len(PERSONAL_SEEDS)
    assert apply_pack(ep, "u", "ent_self", PERSONAL_SEEDS) == 0     # nothing re-added
    assert {b.branch for b in ep.branches("u", "ent_self")} >= {"works_at", "lives_in"}


def test_common_verbs_hit_seed_aliases_without_naming(tmp_path):
    mem, _, ep, llm = _seeded_mem(tmp_path, [
        {"events": [_ev("joined", "Beta Inc", "organization")]},
        {"events": [_ev("took up", "rock climbing", "activity")]},
    ])
    mem.add("u", "I joined Beta Inc.")
    mem.add("u", "I took up rock climbing.")
    self_e = ep.entity_by_alias("u", "the user")
    assert [e.value for e in ep.events("u", entity_id=self_e.entity_id, branch="works_at")] == ["Beta Inc"]
    assert [e.value for e in ep.events("u", entity_id=self_e.entity_id, branch="practices")] == ["rock climbing"]
    assert llm.calls["branch_name"] == 0                   # namer never fired


def test_general_purpose_default_seeds_nothing(tmp_path):
    """No ONTOLOGY_PACK -> empty registry; the ontology emerges only from the
    user's own messages (general-purpose positioning)."""
    facts = JsonMemoryStore(path=str(tmp_path / "facts.json"))
    ep = JsonEpisodicStore(path=str(tmp_path / "ep.json"))
    llm = ScriptedLLM(responses={
        "extract": [{"events": [_ev("joined", "Beta Inc", "organization")]}],
        "branch_name": [{"branch": "works_at", "state_template": "works at",
                         "description": "employment"}]})
    mem = EpisodicMemory(facts, ep, llm, FakeEmbedder(), branch_auto=0.8,
                         branch_gray_low=0.4)  # ontology_pack="" (default)
    mem.add("u", "I joined Beta Inc.")
    self_e = ep.entity_by_alias("u", "the user")
    assert {b.branch for b in ep.branches("u", self_e.entity_id)} == {"works_at"}


def test_registry_feedback_helpers():
    from atomir.episodic.models import BranchRecord
    from atomir.ontology import branch_vocab, format_registry
    bs = [BranchRecord(branch="works_at", user_id="u", entity_id="e",
                       description="employment", state_template="works at",
                       object_type="organization")]
    assert branch_vocab(bs) == ["works_at"]
    assert "works_at: employment [organization]" in format_registry(bs)


def test_novel_predicate_still_creates_new_branch(tmp_path):
    mem, _, ep, llm = _seeded_mem(
        tmp_path, [{"events": [_ev("patented", "a gadget", "object")]}],
        branch_names=[{"branch": "patented", "state_template": "patented",
                       "description": "invention"}])
    mem.add("u", "I patented a gadget.")
    self_e = ep.entity_by_alias("u", "the user")
    assert ep.get_branch("u", self_e.entity_id, "patented") is not None
    assert llm.calls["branch_name"] == 1                   # namer fired for the novel verb
