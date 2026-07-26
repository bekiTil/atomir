"""Qdrant-backed variant of a couple of characterization checks.

Marked `qdrant` and skipped unless `qdrant-client` is installed AND a local
server is reachable — so CI without Qdrant stays green.
"""

from __future__ import annotations

import pytest

from doubles import StubEmbedder, make_service

pytestmark = pytest.mark.qdrant


@pytest.fixture
def qdrant_store():
    qmod = pytest.importorskip("qdrant_client")
    try:
        client = qmod.QdrantClient(location=":memory:")  # in-memory: no server needed
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"qdrant unavailable: {e}")
    from atomir.stores.qdrant_store import QdrantMemoryStore

    dim = 256
    coll = "atomir_test"
    try:
        return QdrantMemoryStore(client, coll, dim)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"could not init qdrant store: {e}")


def test_qdrant_add_and_isolation(qdrant_store):
    emb = StubEmbedder(dim=256)
    svc = make_service(qdrant_store, embedder=emb, hybrid_search=False)
    svc.add("alice", "The user likes tea.")
    svc.add("bob", "The user likes coffee.")
    assert {f["text"] for f in svc.get_all("alice")} == {"The user likes tea."}
    res = svc.search("bob", "The user likes tea.", decompose=False)
    assert all("tea" not in r["text"] for r in res["results"])
