"""Relative-best branch resolution."""

from __future__ import annotations

from atomir.episodic.json_store import JsonEpisodicStore
from atomir.episodic.models import BranchRecord
from atomir.episodic.read import _resolve_branch
from doubles import StubEmbedder


def _store(tmp_path):
    return JsonEpisodicStore(path=str(tmp_path / "ep.json"))


def _branch(store, canonical, aliases, obj="organization"):
    store.add_branch(BranchRecord(branch=canonical, user_id="u", entity_id="ent",
                                  description=f"{canonical} desc", state_template=canonical,
                                  object_type=obj, aliases=aliases))


def test_exact_vocab_lookup_short_circuits(tmp_path):
    s = _store(tmp_path)
    _branch(s, "works_at", ["joined"])
    # Exact canonical match -> resolves without embedding (embedder=None).
    assert _resolve_branch(s, "u", "ent", "works_at", embedder=None).branch == "works_at"
    assert _resolve_branch(s, "u", "ent", "joined", embedder=None).branch == "works_at"


def test_clear_winner_resolves(tmp_path):
    s = _store(tmp_path)
    _branch(s, "works_at", ["joined"])
    _branch(s, "lives_in", ["moved to"], obj="place")
    # StubEmbedder: the hint's rep-text group aligns with works_at only.
    emb = StubEmbedder(synonyms=[["works at", "works_at works_at works_at desc joined"]])
    # A non-vocab hint "works at" — works_at rep scores 1.0, lives_in ~0 -> clear winner.
    b = _resolve_branch(s, "u", "ent", "works at", embedder=emb, floor=0.30, margin=0.10)
    assert b is not None and b.branch == "works_at"


def test_two_close_branches_stay_unresolved(tmp_path):
    s = _store(tmp_path)
    _branch(s, "practices", ["took up"], obj="activity")
    _branch(s, "plays", ["plays"], obj="activity")
    # StubEmbedder makes the hint identical to BOTH reps (cosine 1.0 each) -> no
    # margin -> unresolved.
    reps = ["practices practices practices desc took up", "plays plays plays desc plays"]
    emb = StubEmbedder(synonyms=[["hobby", *reps]])
    assert _resolve_branch(s, "u", "ent", "hobby", embedder=emb, floor=0.30, margin=0.10) is None


def test_empty_registry_is_safe(tmp_path):
    s = _store(tmp_path)
    assert _resolve_branch(s, "u", "ent", "works_at", embedder=StubEmbedder()) is None
    assert _resolve_branch(s, "u", "ent", None, embedder=StubEmbedder()) is None
