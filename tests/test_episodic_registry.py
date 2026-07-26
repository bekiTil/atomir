"""Entity resolver + three-zone branch matcher (Fixes 2, 3, 5)."""

from __future__ import annotations

from atomir.episodic.json_store import JsonEpisodicStore
from atomir.episodic.models import BranchRecord
from atomir.episodic.registry import BranchMatcher, EntityResolver, strip_names
from doubles import ScriptedLLM, StubEmbedder


def _store(tmp_path):
    return JsonEpisodicStore(path=str(tmp_path / "ep.json"))


def _works_at(store, aliases=None):
    return store.add_branch(BranchRecord(
        branch="works_at", user_id="u", entity_id="ent_1",
        description="employment", state_template="works at", aliases=aliases or []))


# --- entity resolution -----------------------------------------------------

def test_entity_resolver_creates_then_reuses(tmp_path):
    r = EntityResolver(_store(tmp_path))
    e1, created1 = r.resolve("u", "Dana")
    assert created1 is True and e1.aliases == ["Dana"]
    e2, created2 = r.resolve("u", "dana")  # casefold reuse
    assert created2 is False and e2.entity_id == e1.entity_id
    assert len(r.store.entities("u")) == 1


def test_entity_resolver_under_merges_by_design(tmp_path):
    """v1 exact matching: 'D' does NOT resolve to 'Dana' on a cold
    store; it fragments into a second entity. Documented known behavior."""
    r = EntityResolver(_store(tmp_path))
    dana, _ = r.resolve("u", "Dana")
    d, created = r.resolve("u", "D")
    assert created is True
    assert d.entity_id != dana.entity_id
    assert len(r.store.entities("u")) == 2  # fragmentation, not a merge


def test_entity_v2_resolves_ambiguous_mention_when_llm_confirms(tmp_path):
    """: with v2 on, 'D' embeds close to 'Dana' and the LLM confirms ->
    resolves to the existing entity instead of fragmenting."""
    r = EntityResolver(_store(tmp_path), StubEmbedder(synonyms=[["D", "Dana"]]),
                       ScriptedLLM(responses={"entity_judge": [{"same": True}]}),
                       v2=True, match_min=0.6)
    dana, _ = r.resolve("u", "Dana")
    d, created = r.resolve("u", "D")
    assert created is False and d.entity_id == dana.entity_id
    assert "D" in r.store.get_entity("u", dana.entity_id).aliases
    assert len(r.store.entities("u")) == 1


def test_entity_v2_keeps_distinct_when_llm_says_different(tmp_path):
    """Two people who share a name must NOT merge, even if embeddings are close."""
    r = EntityResolver(_store(tmp_path), StubEmbedder(synonyms=[["Dana", "Dana Smith"]]),
                       ScriptedLLM(responses={"entity_judge": [{"same": False}]}),
                       v2=True, match_min=0.6)
    r.resolve("u", "Dana")
    _, created = r.resolve("u", "Dana Smith")
    assert created is True and len(r.store.entities("u")) == 2


def test_entity_v2_below_threshold_skips_llm(tmp_path):
    """Dissimilar mention -> new entity without an LLM call."""
    llm = ScriptedLLM(responses={"entity_judge": [{"same": True}]})
    r = EntityResolver(_store(tmp_path), StubEmbedder(), llm, v2=True, match_min=0.6)
    r.resolve("u", "Dana")
    _, created = r.resolve("u", "Xavier")   # orthogonal -> below 0.6
    assert created is True
    assert llm.calls["entity_judge"] == 0   # judge never consulted


# --- strip_names ---------------------------------------------------

def test_strip_names_removes_entity_names_case_insensitive():
    assert strip_names("reports to Dana", {"Dana"}) == "reports to"
    assert strip_names("is friends with Dana", {"Dana"}) == "is friends with"
    assert strip_names("JOINED Acme Corp", {"Acme Corp"}) == "JOINED"
    assert strip_names("left", {"Beta"}) == "left"  # nothing to strip


# --- branch matcher zones --------------------------------------------------

def test_branch_new_when_no_existing_branch(tmp_path):
    store = _store(tmp_path)
    llm = ScriptedLLM(responses={"branch_name": [
        {"branch": "works_at", "state_template": "works at", "description": "employment"}]})
    m = BranchMatcher(store, llm, StubEmbedder(), auto=0.8, gray_low=0.4)
    res = m.match("u", "ent_1", "joined", "person", "organization", {"Acme Corp"})
    assert res["created"] is True and res["zone"] == "new"
    b = res["branch"]
    assert b.branch == "works_at" and b.state_template == "works at"
    assert "joined" in b.aliases
    assert store.get_branch("u", "ent_1", "works_at") is not None


def test_branch_object_type_guard_blocks_cross_type_merge(tmp_path):
    """A person->city 'lives in' must never merge into a person->organization
    'works at', regardless of what the judge says (weak-model over-merge guard)."""
    store = _store(tmp_path)
    store.add_branch(BranchRecord(branch="works_at", user_id="u", entity_id="ent_1",
                                  description="employment", state_template="works at",
                                  object_type="organization"))
    # A liberal judge would wrongly merge; the type guard removes it from the
    # candidate set, so no judge call happens and a new branch is created.
    llm = ScriptedLLM(responses={"branch_judge": [{"branch": "works_at"}],
                                 "branch_name": [{"branch": "lives_in",
                                                  "state_template": "lives in",
                                                  "description": "residence"}]})
    m = BranchMatcher(store, llm, StubEmbedder(), auto=0.8, gray_low=0.4)
    res = m.match("u", "ent_1", "lives in", "person", "city", {"Portland"})
    assert res["created"] is True and res["branch"].branch == "lives_in"
    assert llm.calls["branch_judge"] == 0   # cross-type candidate never reached the judge


def test_branch_exact_alias_match_no_embedding(tmp_path):
    store = _store(tmp_path)
    _works_at(store, aliases=["joined"])
    m = BranchMatcher(store, ScriptedLLM(), StubEmbedder(), auto=0.8, gray_low=0.4)
    res = m.match("u", "ent_1", "joined", "person", "organization", set())
    assert res["zone"] == "exact"
    assert res["judge_used"] is False and res["embedding_calls"] == 0
    assert res["branch"].branch == "works_at"


def test_branch_auto_assign_by_embedding(tmp_path):
    store = _store(tmp_path)
    _works_at(store)
    # Make the candidate embed-identical to the branch's canonical rep text.
    emb = StubEmbedder(synonyms=[[
        "started at (person -> organization)",
        "works at (person -> organization)",
    ]])
    m = BranchMatcher(store, ScriptedLLM(), emb, auto=0.8, gray_low=0.4)
    res = m.match("u", "ent_1", "started at", "person", "organization", {"Acme"})
    assert res["zone"] == "auto" and res["created"] is False
    assert res["judge_used"] is False
    assert "started at" in store.get_branch("u", "ent_1", "works_at").aliases


def test_branch_gray_zone_defers_to_judge(tmp_path):
    store = _store(tmp_path)
    _works_at(store)
    emb = StubEmbedder(synonyms=[[
        "started at (person -> organization)",
        "works at (person -> organization)",
    ]])
    # auto=1.1 forces even a cosine-1.0 match into the gray band -> judge.
    llm = ScriptedLLM(responses={"branch_judge": [{"branch": "works_at"}]})
    m = BranchMatcher(store, llm, emb, auto=1.1, gray_low=0.4)
    res = m.match("u", "ent_1", "started at", "person", "organization", {"Acme"})
    assert res["zone"] == "judge" and res["judge_used"] is True
    assert res["branch"].branch == "works_at"
    assert llm.calls["branch_judge"] == 1


def test_branch_below_gray_still_judged_then_new(tmp_path):
    store = _store(tmp_path)
    _works_at(store)  # existing, but candidate is unrelated (cosine 0 -> below gray)
    llm = ScriptedLLM(responses={
        "branch_judge": [{"branch": "NEW"}],
        "branch_name": [{"branch": "lives_in", "state_template": "lives in",
                         "description": "residence"}],
    })
    m = BranchMatcher(store, llm, StubEmbedder(), auto=0.8, gray_low=0.4)
    res = m.match("u", "ent_1", "moved to", "person", "city", {"Paris"})
    assert res["judge_used"] is True and res["created"] is True
    assert res["branch"].branch == "lives_in"
    assert llm.calls["branch_judge"] == 1 and llm.calls["branch_name"] == 1
