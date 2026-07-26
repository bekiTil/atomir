"""Namer v2: canonical is the knowledge-graph predicate, not the surface verb,
with aliases that include the surface verb."""

from __future__ import annotations

from atomir.embeddings.fake import FakeEmbedder
from atomir.episodic.json_store import JsonEpisodicStore
from atomir.episodic.read import _resolve_branch
from atomir.episodic.registry import BranchMatcher
from doubles import ScriptedLLM


def test_namer_produces_kg_predicate_and_surface_alias(tmp_path):
    store = JsonEpisodicStore(path=str(tmp_path / "ep.json"))
    # Namer maps the surface verb "joined" to the KG predicate works_at, aliases
    # include the surface verb.
    llm = ScriptedLLM(responses={"branch_name": [{
        "branch": "works_at", "state_template": "works at",
        "description": "employment relationship", "aliases": ["joined", "left", "quit"]}]})
    m = BranchMatcher(store, llm, FakeEmbedder(), auto=0.8, gray_low=0.4)
    res = m.match("u", "ent", "joined", "person", "organization", {"Beta Inc"})

    b = res["branch"]
    assert b.branch == "works_at"                     # KG predicate, not "joined"
    assert "joined" in b.aliases                      # surface verb kept as alias
    assert {"left", "quit"} <= set(b.aliases)

    # The planner's natural hint "works_at" now EXACT-matches -> chain walk fires.
    assert _resolve_branch(store, "u", "ent", "works_at", embedder=None).branch == "works_at"
