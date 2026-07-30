"""Engine facade: one clean object the API and SDK call.

`MemoryService` is injected with a `MemoryStore`, an `LLM`, and an `Embedder`
(built elsewhere by their factories). It orchestrates the write and read engines
but imports NO concrete provider or backend — which store, model, or embedder is
used is purely a matter of which instances you inject.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from atomir.atomic_read import atomic_search
from atomir.extractor import extract_facts
from atomir.locking import KeyedLock
from atomir.providers.embedder_base import Embedder
from atomir.providers.llm_base import LLM
from atomir.reconciler import reconcile
from atomir.store_base import MemoryStore

if TYPE_CHECKING:
    from atomir.episodic.engine import EpisodicMemory

_COMPOSE_SYSTEM = (
    "Answer the user's question using ONLY the facts provided. If the facts do "
    "not contain the answer, say you don't know — never invent details. Be concise."
)


class MemoryService:
    """Stable seam over the atomic write/read engines. Backend-injected."""

    def __init__(
        self,
        store: MemoryStore,
        llm: LLM,
        embedder: Embedder,
        *,
        reconcile_min_sim: float = 0.6,
        hybrid_search: bool = True,
        cache_plans: bool = True,
        episodic: "EpisodicMemory | None" = None,
    ) -> None:
        self.store = store
        self.llm = llm
        self.embedder = embedder
        self.reconcile_min_sim = reconcile_min_sim
        self.hybrid_search = hybrid_search
        self.cache_plans = cache_plans
        # When set (EPISODIC_ENABLED), write/read delegate to the event-log
        # engine. When None (default), behavior is exactly the current release.
        self.episodic = episodic
        # Serializes a single user's writes (reconcile is read-modify-write);
        # DECISION #5: simple per-user lock now, full transactions deferred.
        self._locks = KeyedLock()

    def add(self, user_id: str, text: str, recorded_at: str | None = None) -> dict:
        """Extract atomic facts from `text` and reconcile each into memory.

        `recorded_at` (ISO-8601) sets the message time for the episodic layer so
        events anchor correctly (e.g. dated benchmark sessions); ignored by the
        classic fact path, which is timeless. Defaults to now."""
        if self.episodic is not None:
            with self._locks.get(user_id):
                return self.episodic.add(user_id, text, recorded_at=recorded_at)
        operations: list[dict] = []
        facts: list[dict] = []
        # Per-user lock so concurrent adds for the SAME user can't both read the
        # old state and both ADD; different users proceed in parallel.
        with self._locks.get(user_id):
            for candidate in extract_facts(self.llm, text):
                decision = reconcile(
                    self.store,
                    self.llm,
                    self.embedder,
                    user_id,
                    candidate,
                    min_sim=self.reconcile_min_sim,
                )
                operations.append(decision)
                if decision.get("fact"):
                    facts.append(decision["fact"])
        return {"operations": operations, "facts": facts}

    def search(
        self, user_id: str, query: str, k: int = 6, decompose: bool = True
    ) -> dict:
        """Retrieve facts for a query atomically: {subquestions, results}."""
        if self.episodic is not None:
            return self.episodic.search(user_id, query, k=k, decompose=decompose,
                                        hybrid=self.hybrid_search)
        return atomic_search(
            self.store, self.llm, self.embedder, user_id, query, k=k,
            decompose=decompose, hybrid=self.hybrid_search,
            cache_plans=self.cache_plans,
        )

    def timeline(self, user_id: str, entity: str | None = None,
                 branch: str | None = None, since: str | None = None,
                 until: str | None = None) -> list[dict]:
        """Ordered events for an entity/branch. Empty unless episodic is on."""
        if self.episodic is None:
            return []
        return self.episodic.timeline(user_id, entity=entity, branch=branch,
                                      since=since, until=until)

    def answer(
        self, user_id: str, query: str, k: int = 6, decompose: bool = True
    ) -> dict:
        """Retrieve, then COMPOSE a final answer from the facts using the LLM.

        Returns {answer, subquestions, results}. Grounded: the LLM is told to use
        only the retrieved facts and to say it doesn't know otherwise. `search`
        (facts only) remains the default; this is the opt-in composed variant.
        """
        found = self.search(user_id, query, k=k, decompose=decompose)
        context = "; ".join(r["text"] for r in found["results"]) or "(no relevant facts)"
        answer = self.llm.chat_text(_COMPOSE_SYSTEM, f"FACTS: {context}\nQUESTION: {query}")
        return {"answer": answer, "subquestions": found["subquestions"], "results": found["results"]}

    def get_all(self, user_id: str) -> list[dict]:
        return self.store.all(user_id)

    def delete(self, user_id: str, fact_id: str) -> bool:
        with self._locks.get(user_id):
            if self.episodic is not None:
                return self.episodic.delete_fact(user_id, fact_id)  # cascades
            return self.store.delete(user_id, fact_id)

    def forget(self, user_id: str, entity: str) -> bool:
        """Forget everything about an entity by name (cascades across facts,
        events, and episodes). Requires episodic; no-op otherwise."""
        if self.episodic is None:
            return False
        with self._locks.get(user_id):
            rec = self.episodic.episodic.entity_by_alias(user_id, entity)
            if rec is None:
                return False
            return self.episodic.forget_entity(user_id, rec.entity_id)

    def reset(self, user_id: str) -> bool:
        with self._locks.get(user_id):
            if self.episodic is not None:
                return self.episodic.reset(user_id)
            return self.store.clear(user_id)
