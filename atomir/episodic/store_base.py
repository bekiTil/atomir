"""Storage contract for the episodic side: episodes, events, and the entity /
branch registries. Backend-agnostic, per-user like the fact store.

Kept separate from `MemoryStore` (facts) on purpose: facts are a vector store;
episodes/events/registries are relational-ish and need no embeddings. A backend
may implement both (JSON does), or pair a vector store with a SQLite episodic
store.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from atomir.episodic.models import BranchRecord, EntityRecord, Episode, Event


class EpisodicStore(ABC):
    """Abstract, multi-tenant episodic storage. Everything scoped by user_id."""

    # --- episodes ---------------------------------------------------------
    @abstractmethod
    def add_episode(self, episode: Episode) -> Episode: ...
    @abstractmethod
    def get_episode(self, user_id: str, episode_id: str) -> Episode | None: ...
    @abstractmethod
    def delete_episode(self, user_id: str, episode_id: str) -> bool: ...
    @abstractmethod
    def episodes(self, user_id: str) -> list[Episode]:
        """All raw episodes for the user (for semantic recall over original wording)."""

    # --- events (append-mostly log) --------------------------------------
    @abstractmethod
    def append_event(self, event: Event) -> Event: ...
    @abstractmethod
    def get_event(self, user_id: str, event_id: str) -> Event | None: ...
    @abstractmethod
    def update_event(self, event: Event) -> Event | None: ...
    @abstractmethod
    def delete_event(self, user_id: str, event_id: str) -> bool: ...
    @abstractmethod
    def events(
        self,
        user_id: str,
        *,
        entity_id: str | None = None,
        branch: str | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> list[Event]:
        """Events for the user, ordered by occurred_at (fallback recorded_at)."""

    @abstractmethod
    def unprojected_events(self, user_id: str) -> list[Event]:
        """Events with projected=False, in append order (for replay)."""

    # --- entity registry --------------------------------------------------
    @abstractmethod
    def add_entity(self, entity: EntityRecord) -> EntityRecord: ...
    @abstractmethod
    def get_entity(self, user_id: str, entity_id: str) -> EntityRecord | None: ...
    @abstractmethod
    def entity_by_alias(self, user_id: str, name: str) -> EntityRecord | None:
        """Exact casefold alias match (v1 resolution); None if no match."""
    @abstractmethod
    def entities(self, user_id: str) -> list[EntityRecord]: ...
    @abstractmethod
    def update_entity(self, entity: EntityRecord) -> EntityRecord | None: ...
    @abstractmethod
    def delete_entity(self, user_id: str, entity_id: str) -> bool: ...

    # --- branch registry --------------------------------------------------
    @abstractmethod
    def add_branch(self, branch: BranchRecord) -> BranchRecord: ...
    @abstractmethod
    def get_branch(self, user_id: str, entity_id: str, branch: str) -> BranchRecord | None: ...
    @abstractmethod
    def branches(self, user_id: str, entity_id: str | None = None) -> list[BranchRecord]: ...
    @abstractmethod
    def update_branch(self, branch: BranchRecord) -> BranchRecord | None: ...
    @abstractmethod
    def delete_branch(self, user_id: str, entity_id: str, branch: str) -> bool: ...

    # --- maintenance ------------------------------------------------------
    @abstractmethod
    def clear(self, user_id: str) -> bool:
        """Remove ALL episodic data for the user. True if anything existed."""
