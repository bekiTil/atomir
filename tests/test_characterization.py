"""Characterization tests: pin current (v0.7.0) behavior of the public
MemoryService surface with fake providers + the JSON store.

These are the reference for the episodic update's `EPISODIC_ENABLED=false`
regression guarantee: after, this suite MUST pass
unchanged with the flag off. They assert what the code does TODAY, not what it
ideally should — that's the point of characterization.
"""

from __future__ import annotations

from atomir.embeddings.fake import FakeEmbedder
from atomir.llm.fake import FakeLLM
from doubles import ScriptedLLM, StubEmbedder, first_neighbor_id, make_service

MANAGER_DANA = "The user's manager is Dana."
MANAGER_SAM = "The user's manager is Sam."


# --- add / reconcile -------------------------------------------------------

def test_add_default_extraction_splits_and_adds(service):
    """Default FakeLLM naive-splits sentences; distinct facts each ADD (the
    similarity gate short-circuits before the reconcile LLM)."""
    res = service.add("u1", "The user is vegetarian. The user's manager is Dana.")
    assert [op["decision"] for op in res["operations"]] == ["ADD", "ADD"]
    texts = {f["text"] for f in service.get_all("u1")}
    assert texts == {"The user is vegetarian.", MANAGER_DANA}


def test_add_reconcile_update_supersedes_with_history():
    """Same-attribute candidate (gate opened via StubEmbedder) + scripted UPDATE
    -> one live fact, old text pushed into history."""
    from atomir.stores.json_store import JsonMemoryStore
    import tempfile, os

    path = os.path.join(tempfile.mkdtemp(), "s.json")
    store = JsonMemoryStore(path=path)
    llm = ScriptedLLM(responses={
        "extract": [{"facts": [MANAGER_DANA]}, {"facts": [MANAGER_SAM]}],
        "reconcile": [lambda s, u: {"decision": "UPDATE",
                                    "target_id": first_neighbor_id(u),
                                    "reason": "same attribute"}],
    })
    emb = StubEmbedder(synonyms=[[MANAGER_DANA, MANAGER_SAM]])
    svc = make_service(store, llm=llm, embedder=emb)

    svc.add("u", "msg1")   # ADD Dana (no neighbours -> gate ADD)
    res = svc.add("u", "msg2")  # gate opens (cosine 1.0) -> scripted UPDATE

    assert res["operations"][0]["decision"] == "UPDATE"
    facts = svc.get_all("u")
    assert len(facts) == 1
    assert facts[0]["text"] == MANAGER_SAM
    assert facts[0]["history"] == [MANAGER_DANA]
    assert llm.calls["reconcile"] == 1  # only the 2nd add consulted the LLM


def test_add_reconcile_delete_removes_fact(store):
    llm = ScriptedLLM(responses={
        "extract": [{"facts": ["The user owns a car."]},
                    {"facts": ["The user no longer owns a car."]}],
        "reconcile": [lambda s, u: {"decision": "DELETE",
                                    "target_id": first_neighbor_id(u)}],
    })
    emb = StubEmbedder(synonyms=[["The user owns a car.",
                                  "The user no longer owns a car."]])
    svc = make_service(store, llm=llm, embedder=emb)
    svc.add("u", "m1")
    res = svc.add("u", "m2")
    assert res["operations"][0]["decision"] == "DELETE"
    assert svc.get_all("u") == []


def test_add_reconcile_noop_leaves_fact_unchanged(store):
    llm = ScriptedLLM(responses={
        "extract": [{"facts": ["The user likes tea."]},
                    {"facts": ["The user enjoys tea."]}],
        "reconcile": [{"decision": "NOOP", "target_id": None}],
    })
    emb = StubEmbedder(synonyms=[["The user likes tea.", "The user enjoys tea."]])
    svc = make_service(store, llm=llm, embedder=emb)
    svc.add("u", "m1")
    res = svc.add("u", "m2")
    assert res["operations"][0]["decision"] == "NOOP"
    facts = svc.get_all("u")
    assert len(facts) == 1 and facts[0]["text"] == "The user likes tea."


# --- search ----------------------------------------------------------------

def test_search_dense_exact_match_ranks_first(store):
    svc = make_service(store, hybrid_search=False)
    svc.add("u", "The user is vegetarian. The user's manager is Dana.")
    res = svc.search("u", MANAGER_DANA, decompose=False)
    assert res["subquestions"] == [MANAGER_DANA]
    assert res["results"][0]["text"] == MANAGER_DANA


def test_search_decompose_off_uses_raw_query(service):
    service.add("u", "The user has a dog.")
    res = service.search("u", "anything at all?", decompose=False)
    assert res["subquestions"] == ["anything at all?"]


def test_search_decompose_unions_subquestions(store):
    """Scripted planner returns two sub-questions; each retrieves its fact; the
    union contains both."""
    llm = ScriptedLLM(responses={
        "extract": [{"facts": ["FACT A"]}, {"facts": ["FACT B"]}],
        "plan": [{"decompose": True, "subquestions": ["Q1", "Q2"]}],
    })
    emb = StubEmbedder(synonyms=[["Q1", "FACT A"], ["Q2", "FACT B"]])
    svc = make_service(store, llm=llm, embedder=emb, hybrid_search=False)
    svc.add("u", "m1")
    svc.add("u", "m2")
    res = svc.search("u", "compound question", decompose=True)
    assert res["subquestions"] == ["Q1", "Q2"]
    assert {r["text"] for r in res["results"]} == {"FACT A", "FACT B"}


def test_hybrid_recovers_lexical_match_that_dense_misses(store):
    """With the query orthogonal to every fact vector, dense (k=3) misses the
    last-inserted 'Apollo' fact; hybrid's BM25 lexical ranking recovers it."""
    emb = StubEmbedder()  # one shared instance: every query is orthogonal to all facts
    facts = ["The user likes tea.", "The user has a dog.", "The user drives a sedan.",
             "The user plays guitar.", "The user's project is Apollo."]
    dense = make_service(store, embedder=emb, hybrid_search=False)
    for f in facts:
        dense.add("u", f)
    hybrid = make_service(store, embedder=emb, hybrid_search=True)

    dense_texts = " ".join(r["text"] for r in
                           dense.search("u", "Apollo", k=3, decompose=False)["results"])
    hybrid_texts = " ".join(r["text"] for r in
                            hybrid.search("u", "Apollo", k=3, decompose=False)["results"])
    assert "Apollo" not in dense_texts     # dense alone misses it
    assert "Apollo" in hybrid_texts        # hybrid recovers it


# --- answer -----------------------------------------------------------------

def test_answer_composes_from_retrieved_facts(store):
    llm = ScriptedLLM(responses={
        "extract": [{"facts": [MANAGER_DANA]}],
        "text": ["Composed: Dana."],
    })
    svc = make_service(store, llm=llm, embedder=FakeEmbedder())
    svc.add("u", "m1")
    out = svc.answer("u", MANAGER_DANA, decompose=False)
    assert out["answer"] == "Composed: Dana."
    assert out["results"][0]["text"] == MANAGER_DANA
    assert "subquestions" in out


# --- get_all / delete / reset / isolation ----------------------------------

def test_get_all_scoped_per_user(service):
    service.add("alice", "The user likes tea.")
    service.add("bob", "The user likes coffee.")
    assert {f["text"] for f in service.get_all("alice")} == {"The user likes tea."}
    assert {f["text"] for f in service.get_all("bob")} == {"The user likes coffee."}


def test_delete_returns_bool_and_removes(service):
    service.add("u", "The user has a dog.")
    fid = service.get_all("u")[0]["id"]
    assert service.delete("u", fid) is True
    assert service.get_all("u") == []
    assert service.delete("u", "does-not-exist") is False


def test_reset_clears_only_that_user(service):
    service.add("alice", "The user likes tea.")
    service.add("bob", "The user likes coffee.")
    assert service.reset("alice") is True
    assert service.get_all("alice") == []
    assert len(service.get_all("bob")) == 1
    assert service.reset("nobody") is False


def test_search_isolated_across_users(store):
    svc = make_service(store, hybrid_search=False)
    svc.add("alice", MANAGER_DANA)
    svc.add("bob", "The user has a dog.")
    res = svc.search("bob", MANAGER_DANA, decompose=False)
    assert all("Dana" not in r["text"] for r in res["results"])


# --- persistence -----------------------------------------------------------

def test_json_store_persists_across_reopen(tmp_path):
    from atomir.stores.json_store import JsonMemoryStore

    path = str(tmp_path / "persist.json")
    svc = make_service(JsonMemoryStore(path=path))
    svc.add("u", "The user has a dog.")
    # Re-open a fresh store over the same file: facts survive.
    reopened = make_service(JsonMemoryStore(path=path))
    assert {f["text"] for f in reopened.get_all("u")} == {"The user has a dog."}
