"""Reference `EpisodicStore`: one JSON file with four sections.

Mirrors `JsonMemoryStore`'s durability (temp file + fsync + atomic os.replace)
and its "dev / small scale only" caveat — append-heavy episodic use accelerates
that limit. Events are stored flat and ordered on read.
"""

from __future__ import annotations

import json
import os
import tempfile

from atomir.episodic.models import BranchRecord, EntityRecord, Episode, Event
from atomir.episodic.store_base import EpisodicStore

_EMPTY: dict[str, list] = {"episodes": [], "events": [], "entities": [], "branches": []}


class JsonEpisodicStore(EpisodicStore):
    def __init__(self, path: str = "./atomir_episodic.json") -> None:
        self.path = path
        self._db = self._load()

    # --- persistence ------------------------------------------------------
    def _load(self) -> dict:
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for key in _EMPTY:
                data.setdefault(key, [])
            return data
        return {k: [] for k in _EMPTY}

    def _save(self) -> None:
        directory = os.path.dirname(os.path.abspath(self.path)) or "."
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".atomir-ep-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._db, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # --- episodes ---------------------------------------------------------
    def add_episode(self, episode: Episode) -> Episode:
        self._db["episodes"].append(episode.to_dict())
        self._save()
        return episode

    def get_episode(self, user_id: str, episode_id: str) -> Episode | None:
        for d in self._db["episodes"]:
            if d["id"] == episode_id and d["user_id"] == user_id:
                return Episode.from_dict(d)
        return None

    def episodes(self, user_id: str) -> list[Episode]:
        return [Episode.from_dict(d) for d in self._db["episodes"] if d["user_id"] == user_id]

    def delete_episode(self, user_id: str, episode_id: str) -> bool:
        before = len(self._db["episodes"])
        self._db["episodes"] = [
            d for d in self._db["episodes"]
            if not (d["id"] == episode_id and d["user_id"] == user_id)
        ]
        if len(self._db["episodes"]) != before:
            self._save()
            return True
        return False

    # --- events -----------------------------------------------------------
    def append_event(self, event: Event) -> Event:
        self._db["events"].append(event.to_dict())
        self._save()
        return event

    def get_event(self, user_id: str, event_id: str) -> Event | None:
        for d in self._db["events"]:
            if d["id"] == event_id and d["user_id"] == user_id:
                return Event.from_dict(d)
        return None

    def update_event(self, event: Event) -> Event | None:
        for i, d in enumerate(self._db["events"]):
            if d["id"] == event.id and d["user_id"] == event.user_id:
                self._db["events"][i] = event.to_dict()
                self._save()
                return event
        return None

    def delete_event(self, user_id: str, event_id: str) -> bool:
        before = len(self._db["events"])
        self._db["events"] = [
            d for d in self._db["events"]
            if not (d["id"] == event_id and d["user_id"] == user_id)
        ]
        if len(self._db["events"]) != before:
            self._save()
            return True
        return False

    def events(
        self,
        user_id: str,
        *,
        entity_id: str | None = None,
        branch: str | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> list[Event]:
        out = [Event.from_dict(d) for d in self._db["events"] if d["user_id"] == user_id]
        if entity_id is not None:
            out = [e for e in out if e.entity_id == entity_id]
        if branch is not None:
            out = [e for e in out if e.branch == branch]
        if since is not None:
            out = [e for e in out if e.order_key >= since]
        if until is not None:
            out = [e for e in out if e.order_key <= until]
        out.sort(key=lambda e: e.order_key)
        return out

    def unprojected_events(self, user_id: str) -> list[Event]:
        return [
            Event.from_dict(d) for d in self._db["events"]
            if d["user_id"] == user_id and not d.get("projected", False)
        ]

    # --- entity registry --------------------------------------------------
    def add_entity(self, entity: EntityRecord) -> EntityRecord:
        self._db["entities"].append(entity.to_dict())
        self._save()
        return entity

    def get_entity(self, user_id: str, entity_id: str) -> EntityRecord | None:
        for d in self._db["entities"]:
            if d["entity_id"] == entity_id and d["user_id"] == user_id:
                return EntityRecord.from_dict(d)
        return None

    def entity_by_alias(self, user_id: str, name: str) -> EntityRecord | None:
        needle = name.casefold().strip()
        for d in self._db["entities"]:
            if d["user_id"] != user_id:
                continue
            if any(a.casefold().strip() == needle for a in d.get("aliases", [])):
                return EntityRecord.from_dict(d)
        return None

    def entities(self, user_id: str) -> list[EntityRecord]:
        return [EntityRecord.from_dict(d) for d in self._db["entities"] if d["user_id"] == user_id]

    def update_entity(self, entity: EntityRecord) -> EntityRecord | None:
        for i, d in enumerate(self._db["entities"]):
            if d["entity_id"] == entity.entity_id and d["user_id"] == entity.user_id:
                self._db["entities"][i] = entity.to_dict()
                self._save()
                return entity
        return None

    def delete_entity(self, user_id: str, entity_id: str) -> bool:
        before = len(self._db["entities"])
        self._db["entities"] = [
            d for d in self._db["entities"]
            if not (d["entity_id"] == entity_id and d["user_id"] == user_id)
        ]
        if len(self._db["entities"]) != before:
            self._save()
            return True
        return False

    # --- branch registry --------------------------------------------------
    def add_branch(self, branch: BranchRecord) -> BranchRecord:
        self._db["branches"].append(branch.to_dict())
        self._save()
        return branch

    def get_branch(self, user_id: str, entity_id: str, branch: str) -> BranchRecord | None:
        for d in self._db["branches"]:
            if d["user_id"] == user_id and d["entity_id"] == entity_id and d["branch"] == branch:
                return BranchRecord.from_dict(d)
        return None

    def branches(self, user_id: str, entity_id: str | None = None) -> list[BranchRecord]:
        out = [BranchRecord.from_dict(d) for d in self._db["branches"] if d["user_id"] == user_id]
        if entity_id is not None:
            out = [b for b in out if b.entity_id == entity_id]
        return out

    def update_branch(self, branch: BranchRecord) -> BranchRecord | None:
        for i, d in enumerate(self._db["branches"]):
            if (d["user_id"] == branch.user_id and d["entity_id"] == branch.entity_id
                    and d["branch"] == branch.branch):
                self._db["branches"][i] = branch.to_dict()
                self._save()
                return branch
        return None

    def delete_branch(self, user_id: str, entity_id: str, branch: str) -> bool:
        before = len(self._db["branches"])
        self._db["branches"] = [
            d for d in self._db["branches"]
            if not (d["user_id"] == user_id and d["entity_id"] == entity_id
                    and d["branch"] == branch)
        ]
        if len(self._db["branches"]) != before:
            self._save()
            return True
        return False

    # --- maintenance ------------------------------------------------------
    def clear(self, user_id: str) -> bool:
        existed = False
        for key in _EMPTY:
            before = len(self._db[key])
            self._db[key] = [d for d in self._db[key] if d["user_id"] != user_id]
            existed = existed or len(self._db[key]) != before
        if existed:
            self._save()
        return existed
