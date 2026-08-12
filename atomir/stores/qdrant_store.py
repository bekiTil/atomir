"""Production `MemoryStore` backend backed by a real vector DB (Qdrant example).

Qdrant is ONE interchangeable example — pgvector, Chroma, Pinecone, Weaviate,
etc. would each be another `MemoryStore`; the engine never knows which. This
module top-level-imports the qdrant SDK on purpose: it is only imported when a
caller actually wants Qdrant (offline/fake paths never touch it).

DECISION #3 — isolation by FILTER: one collection, `user_id` in each point's
payload, and every search/all/clear filters on it. No collection-per-user.

The `QdrantClient` is INJECTED so tests can pass `QdrantClient(":memory:")` and
production can pass a server/on-disk client — same code either way.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from qdrant_client import QdrantClient, models

from atomir.store_base import MemoryStore


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _user_filter(user_id: str) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
    )


class QdrantMemoryStore(MemoryStore):
    """MemoryStore over Qdrant. One collection, user_id-filtered isolation."""

    def __init__(self, client: QdrantClient, collection: str, embed_dim: int) -> None:
        self.client = client
        self.collection = collection
        self.embed_dim = embed_dim
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=self.embed_dim, distance=models.Distance.COSINE
                ),
            )

    # --- contract ---------------------------------------------------------
    def add(
        self,
        user_id: str,
        text: str,
        embedding: list[float],
        metadata: dict | None = None,
    ) -> dict:
        now = _now()
        pid = str(uuid4())
        payload = {
            "user_id": user_id,
            "text": text,
            "created_at": now,
            "updated_at": now,
            "history": [],
            "metadata": metadata or {},
        }
        self.client.upsert(
            collection_name=self.collection,
            points=[models.PointStruct(id=pid, vector=list(embedding), payload=payload)],
        )
        return {"id": pid, **payload}

    def search(
        self,
        user_id: str,
        query_embedding: list[float],
        k: int = 5,
    ) -> list[dict]:
        res = self.client.query_points(
            collection_name=self.collection,
            query=list(query_embedding),
            query_filter=_user_filter(user_id),  # isolation
            limit=k,
            with_payload=True,
        ).points
        return [{"id": str(p.id), "score": float(p.score), **p.payload} for p in res]

    def get(self, user_id: str, fact_id: str) -> dict | None:
        pts = self.client.retrieve(
            collection_name=self.collection, ids=[fact_id], with_payload=True
        )
        if not pts:
            return None
        p = pts[0]
        if p.payload.get("user_id") != user_id:  # cross-tenant read guard
            return None
        return {"id": str(p.id), **p.payload}

    def update(
        self,
        user_id: str,
        fact_id: str,
        new_text: str,
        new_embedding: list[float],
        metadata: dict | None = None,
    ) -> dict | None:
        pts = self.client.retrieve(
            collection_name=self.collection, ids=[fact_id], with_payload=True
        )
        if not pts or pts[0].payload.get("user_id") != user_id:
            return None
        payload = dict(pts[0].payload)
        if payload.get("text") != new_text:
            payload["history"] = list(payload.get("history", [])) + [payload["text"]]
        payload["text"] = new_text
        payload["updated_at"] = _now()
        if metadata:
            payload.setdefault("metadata", {}).update(metadata)
        # Re-upsert the SAME id with new vector + payload.
        self.client.upsert(
            collection_name=self.collection,
            points=[models.PointStruct(id=fact_id, vector=list(new_embedding), payload=payload)],
        )
        return {"id": str(fact_id), **payload}

    def delete(self, user_id: str, fact_id: str) -> bool:
        if self.get(user_id, fact_id) is None:  # also enforces ownership
            return False
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.PointIdsList(points=[fact_id]),
        )
        return True

    def all(self, user_id: str, limit: int = 1000) -> list[dict]:
        pts, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=_user_filter(user_id),
            limit=limit,
            with_payload=True,
        )
        return [{"id": str(p.id), **p.payload} for p in pts]

    def clear(self, user_id: str) -> bool:
        existed = bool(self.all(user_id, limit=1))
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(filter=_user_filter(user_id)),
        )
        return existed
