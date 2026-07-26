"""Resolved-chain cap: a resolved branch longer than 3*k
returns checkpoints + top-k, not the whole chain."""

from __future__ import annotations

from atomir.embeddings.fake import FakeEmbedder
from atomir.episodic.engine import EpisodicMemory
from atomir.episodic.json_store import JsonEpisodicStore
from atomir.episodic.models import BranchRecord, EntityRecord, Event, new_id
from atomir.stores.json_store import JsonMemoryStore
from doubles import ScriptedLLM


def test_resolved_chain_capped_to_k(tmp_path):
    facts = JsonMemoryStore(path=str(tmp_path / "facts.json"))
    ep = JsonEpisodicStore(path=str(tmp_path / "ep.json"))
    ep.add_entity(EntityRecord(entity_id="ent_self", user_id="u",
                               canonical_name="the user", aliases=["the user"]))
    ep.add_branch(BranchRecord(branch="works_at", user_id="u", entity_id="ent_self",
                               description="employment", state_template="works at",
                               object_type="organization"))
    for i in range(25):                       # 25 nodes >> 3*k (=18)
        ep.append_event(Event(id=new_id("ev"), user_id="u", entity_id="ent_self",
                              branch="works_at", value=f"Co{i}", raw_text="x",
                              polarity="update", recorded_at=f"2025-02-{i+1:02d}T00:00:00",
                              projected=True))
    llm = ScriptedLLM(responses={"plan": [{"subquestions": [
        {"text": "companies over time", "type": "temporal",
         "entity_hint": "the user", "branch_hint": "works_at"}]}]})
    mem = EpisodicMemory(facts, ep, llm, FakeEmbedder(), branch_auto=0.8, branch_gray_low=0.4)

    res = mem.search("u", "employment history?", k=6)
    assert res["branch_resolved"] is True
    assert res["fallback_used"] is False
    assert len(res["results"]) <= 6           # capped from 25
