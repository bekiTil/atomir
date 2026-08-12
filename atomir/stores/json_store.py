"""Reference `MemoryStore` backend: one JSON file, linear scan, numpy cosine.

The simplest thing that satisfies the Step 2 contract. It exists to (a) prove
the interface is implementable and correct before adding a real DB, and (b) be a
zero-dependency, zero-key backend for local dev and tests. It is demo-grade — a
linear scan over a flat list — and is expected to be swapped for Qdrant (Step 4).

Every method filters by `user_id`: all tenants share one flat list, so isolation
is the store's responsibility on every read, write, and delete.

DURABILITY: `_save` writes to a temp file, fsyncs it, and atomically `os.replace`s
it over the target, so a crash mid-write leaves the previous good file intact
(never a truncated/corrupt one). It is still a single-file, single-process store
(no multi-process locking, whole-file rewrite per save) — fine for dev, small
deployments, and tests; use a real DB backend (e.g. Qdrant) at scale.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np

from atomir.store_base import MemoryStore


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JsonMemoryStore(MemoryStore):
    """A flat-file, in-process store. Persists to a single JSON file."""

    def __init__(self, path: str = "./atomir_store.json") -> None:
        self.path = path
        self._facts: list[dict] = self._load()

    # --- persistence ------------------------------------------------------
    def _load(self) -> list[dict]:
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save(self) -> None:
        # Atomic write: fully write + fsync a temp file in the same directory,
        # then os.replace() it over the target (atomic on POSIX and Windows).
        # A crash mid-write leaves the previous file intact, never a partial one.
        directory = os.path.dirname(os.path.abspath(self.path)) or "."
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".atomir-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._facts, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _public(self, fact: dict) -> dict:
        """A copy without the raw embedding (an internal detail)."""
        return {k: v for k, v in fact.items() if k != "embedding"}

    # --- contract ---------------------------------------------------------
    def add(
        self,
        user_id: str,
        text: str,
        embedding: list[float],
        metadata: dict | None = None,
    ) -> dict:
        now = _now()
        fact = {
            "id": str(uuid4()),
            "user_id": user_id,
            "text": text,
            "embedding": list(embedding),
            "created_at": now,
            "updated_at": now,
            "history": [],
            "metadata": metadata or {},
        }
        self._facts.append(fact)
        self._save()
        return self._public(fact)

    def search(
        self,
        user_id: str,
        query_embedding: list[float],
        k: int = 5,
    ) -> list[dict]:
        q = np.asarray(query_embedding, dtype=float)
        qn = float(np.linalg.norm(q))
        scored: list[dict] = []
        for fact in self._facts:
            if fact["user_id"] != user_id:  # isolation
                continue
            v = np.asarray(fact["embedding"], dtype=float)
            vn = float(np.linalg.norm(v))
            score = float(q.dot(v) / (qn * vn)) if qn and vn else 0.0
            out = self._public(fact)
            out["score"] = score
            scored.append(out)
        scored.sort(key=lambda d: d["score"], reverse=True)
        return scored[:k]

    def get(self, user_id: str, fact_id: str) -> dict | None:
        for fact in self._facts:
            if fact["id"] == fact_id and fact["user_id"] == user_id:
                return self._public(fact)
        return None

    def update(
        self,
        user_id: str,
        fact_id: str,
        new_text: str,
        new_embedding: list[float],
        metadata: dict | None = None,
    ) -> dict | None:
        for fact in self._facts:
            if fact["id"] == fact_id and fact["user_id"] == user_id:
                if fact["text"] != new_text:
                    fact["history"].append(fact["text"])  # keep superseded text
                fact["text"] = new_text
                fact["embedding"] = list(new_embedding)
                fact["updated_at"] = _now()
                if metadata:
                    fact.setdefault("metadata", {}).update(metadata)
                self._save()
                return self._public(fact)
        return None

    def delete(self, user_id: str, fact_id: str) -> bool:
        before = len(self._facts)
        self._facts = [
            f for f in self._facts
            if not (f["id"] == fact_id and f["user_id"] == user_id)
        ]
        if len(self._facts) != before:
            self._save()
            return True
        return False

    def all(self, user_id: str, limit: int = 1000) -> list[dict]:
        out = [self._public(f) for f in self._facts if f["user_id"] == user_id]
        return out[:limit]

    def clear(self, user_id: str) -> bool:
        before = len(self._facts)
        self._facts = [f for f in self._facts if f["user_id"] != user_id]
        if len(self._facts) != before:
            self._save()
            return True
        return False
