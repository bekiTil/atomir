"""Embedding cache: the read path must not re-embed stored events/branches on
every query (the LoCoMo-pilot scaling bug). Pure performance — identical vectors."""

from __future__ import annotations

from atomir.embeddings.fake import FakeEmbedder
from atomir.episodic.engine import EpisodicMemory
from atomir.episodic.json_store import JsonEpisodicStore
from atomir.episodic.models import BranchRecord, EntityRecord, Event, new_id, now_iso
from atomir.episodic.telemetry import CountingEmbedder, MemoizingEmbedder
from atomir.stores.json_store import JsonMemoryStore
from doubles import ScriptedLLM


def test_memoizing_embedder_reuses_vectors():
    counting = CountingEmbedder(FakeEmbedder(embed_dim=16))
    emb = MemoizingEmbedder(counting)
    v1 = emb.embed_passage("The user works at Acme")
    v2 = emb.embed_passage("The user works at Acme")   # cache hit
    assert v1 == v2 and counting.c.embedding == 1      # embedded ONCE, not twice
    emb.embed_query("The user works at Acme")           # query mode = distinct key
    assert counting.c.embedding == 2
    emb.embed_passage("The user lives in Paris")         # new text -> new embed
    assert counting.c.embedding == 3


def test_repeated_search_does_not_re_embed_events(tmp_path):
    facts = JsonMemoryStore(path=str(tmp_path / "f.json"))
    ep = JsonEpisodicStore(path=str(tmp_path / "e.json"))
    ep.add_entity(EntityRecord(entity_id="ent_self", user_id="u",
                               canonical_name="the user", aliases=["the user"]))
    ep.add_branch(BranchRecord(branch="works_at", user_id="u", entity_id="ent_self",
                               description="employment", state_template="works at",
                               object_type="organization"))
    for i in range(20):                                  # 20 events -> fallback embeds each
        ep.append_event(Event(id=new_id("ev"), user_id="u", entity_id="ent_self",
                              branch="works_at", value=f"Co{i}", raw_text="x",
                              polarity="start", recorded_at=now_iso(), projected=True))
    # Hint that won't resolve -> temporal route falls back to ranking every event.
    plan = [{"subquestions": [{"text": "companies over time", "type": "temporal",
                              "entity_hint": "the user", "branch_hint": "zzz_nomatch"}]}]
    llm = ScriptedLLM(responses={"plan": [plan[0], plan[0]]})
    mem = EpisodicMemory(facts, ep, llm, FakeEmbedder(), branch_auto=0.8, branch_gray_low=0.4)

    mem.search("u", "which companies?", k=6)             # 1st query: warms the cache
    warm = mem._emb.c.embedding
    assert warm >= 20                                    # it DID embed the events once
    mem.search("u", "which companies?", k=6)             # identical query
    assert mem._emb.c.embedding == warm                  # 2nd query: ZERO new embeds
