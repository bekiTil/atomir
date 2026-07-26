"""Semantic regression: ON must not lose to OFF because projection state-phrasing
drops a keyword the query matches on. The semantic
route recovers it from the raw episode text."""

from __future__ import annotations

from atomir.embeddings.fake import FakeEmbedder
from atomir.episodic.engine import EpisodicMemory
from atomir.episodic.json_store import JsonEpisodicStore
from atomir.llm.fake import FakeLLM
from atomir.memory import MemoryService
from atomir.stores.json_store import JsonMemoryStore
from doubles import ScriptedLLM

MSG = "I took up landscape painting as a hobby."
Q = "what hobby did the user take up"


def _recall(results, gold):
    blob = " ".join(r["text"] for r in results).casefold()
    return 1.0 if gold.casefold() in blob else 0.0


def test_on_semantic_recovers_dropped_keyword_and_matches_off(tmp_path):
    # --- ON: episodic. Projection -> "The user practices landscape painting"
    # (the word "hobby" is dropped); the raw episode still has "hobby".
    facts_on = JsonMemoryStore(path=str(tmp_path / "on_facts.json"))
    ep = JsonEpisodicStore(path=str(tmp_path / "ep.json"))
    on_llm = ScriptedLLM(responses={
        "extract": [{"events": [{"verb_phrase": "took up", "value": "landscape painting",
                                 "subject": "the user", "subject_type": "person",
                                 "object_type": "activity", "polarity": "start",
                                 "modality": "happened", "occurred_at": None,
                                 "raw_text": "took up landscape painting"}]}],
        "plan": [{"subquestions": [{"text": Q, "type": "semantic",
                                    "entity_hint": None, "branch_hint": None}]}]})
    engine = EpisodicMemory(facts_on, ep, on_llm, FakeEmbedder(), branch_auto=0.8,
                            branch_gray_low=0.4, ontology_pack="personal")
    on = MemoryService(facts_on, on_llm, FakeEmbedder(), episodic=engine)
    on.add("u", MSG)

    # The projected fact really does drop "hobby" (so recall can only come from
    # the episode — the fix is load-bearing).
    assert all("hobby" not in f["text"].casefold() for f in facts_on.all("u"))

    on_res = on.search("u", Q)
    assert any(r.get("source") == "episode" for r in on_res["results"])   # episode recovered
    on_recall = _recall(on_res["results"], "hobby")

    # --- OFF: classic naive extraction keeps the original sentence (has "hobby").
    facts_off = JsonMemoryStore(path=str(tmp_path / "off_facts.json"))
    off = MemoryService(facts_off, FakeLLM(), FakeEmbedder())
    off.add("u", MSG)
    off_recall = _recall(off.search("u", Q)["results"], "hobby")

    assert on_recall == 1.0
    assert on_recall >= off_recall     # no regression vs classic
