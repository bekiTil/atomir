"""The storage contract the engine depends on — backend-agnostic.

`MemoryStore` is an abstract base class: it declares WHAT a store must do, never
HOW. Every backend (the JSON reference store, Qdrant, anything later) implements
exactly this method set, so the engine can talk to any of them identically and
storage becomes swappable by config.

Two decisions (DECISION #2) are baked into the contract:
  1. This minimal method set is the WHOLE contract — the engine may not assume
     any operation not listed here.
  2. `user_id` is the first argument of EVERY method: multi-tenancy is
     structural from day one, not bolted on later.

Fact schema — the dict shape every stored memory follows:
    id          : str   — unique within a user's namespace
    user_id     : str   — tenant this fact belongs to
    text        : str   — the atomic fact itself
    embedding   : list[float]  — the vector (optional in returned dicts)
    created_at  : str   — ISO-8601 timestamp
    updated_at  : str   — ISO-8601 timestamp
    history     : list[str]    — superseded texts, newest last (filled by update)
    metadata    : dict         — caller-supplied extra fields
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class MemoryStore(ABC):
    """Abstract, multi-tenant storage contract. Everything is scoped by user."""

    @abstractmethod
    def add(
        self,
        user_id: str,
        text: str,
        embedding: list[float],
        metadata: dict | None = None,
    ) -> dict:
        """Store a new fact and return it (including its generated `id`)."""

    @abstractmethod
    def search(
        self,
        user_id: str,
        query_embedding: list[float],
        k: int = 5,
    ) -> list[dict]:
        """Return the k most similar facts for the user; each has `id`, `score`, `text`, ..."""

    @abstractmethod
    def get(self, user_id: str, fact_id: str) -> dict | None:
        """Return the fact by id, or None if it does not exist for this user."""

    @abstractmethod
    def update(
        self,
        user_id: str,
        fact_id: str,
        new_text: str,
        new_embedding: list[float],
        metadata: dict | None = None,
    ) -> dict | None:
        """Replace a fact's text/embedding, pushing the old text into `history`.

        Optional `metadata` merges into the fact's existing metadata dict (used
        by append-only projection to flip is_current on a demoted fact without
        touching text/embedding).

        Returns the updated fact, or None if it does not exist for this user.
        """

    @abstractmethod
    def delete(self, user_id: str, fact_id: str) -> bool:
        """Remove a fact. Returns True if something was deleted."""

    @abstractmethod
    def all(self, user_id: str, limit: int = 1000) -> list[dict]:
        """Return up to `limit` of the user's facts."""

    @abstractmethod
    def clear(self, user_id: str) -> bool:
        """Remove all of the user's facts. Returns True if the namespace existed."""
