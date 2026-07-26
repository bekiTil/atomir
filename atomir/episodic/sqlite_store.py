"""SQLite `EpisodicStore`: a relational home for the episodic side,
using stdlib `sqlite3` (no new dependency). Better than the JSON store for the
append-heavy event log — indexed lookups and no whole-file rewrite.

Each record is stored as its JSON dict in a `data` column, alongside the few
columns we filter/order on (kept in sync on write). A process-wide lock
serializes access since a single connection isn't safe for concurrent use;
fine for the single-process, dev-to-moderate scale this targets.
"""

from __future__ import annotations

import json
import sqlite3
import threading

from atomir.episodic.models import BranchRecord, EntityRecord, Episode, Event
from atomir.episodic.store_base import EpisodicStore

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes(user_id TEXT, id TEXT PRIMARY KEY, data TEXT);
CREATE TABLE IF NOT EXISTS events(
  user_id TEXT, id TEXT PRIMARY KEY, entity_id TEXT, branch TEXT,
  projected INTEGER, order_key TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS entities(user_id TEXT, entity_id TEXT PRIMARY KEY, data TEXT);
CREATE TABLE IF NOT EXISTS branches(
  user_id TEXT, entity_id TEXT, branch TEXT, data TEXT,
  PRIMARY KEY(user_id, entity_id, branch));
CREATE INDEX IF NOT EXISTS ev_user ON events(user_id);
CREATE INDEX IF NOT EXISTS ent_user ON entities(user_id);
CREATE INDEX IF NOT EXISTS br_user ON branches(user_id);
"""


class SqliteEpisodicStore(EpisodicStore):
    def __init__(self, path: str = "./atomir_episodic.db") -> None:
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self._lock:
            self.conn.executescript(_SCHEMA)
            self.conn.commit()

    def _exec(self, sql: str, params=()):
        with self._lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur

    def _rows(self, sql: str, params=()):
        with self._lock:
            return self.conn.execute(sql, params).fetchall()

    # --- episodes ---------------------------------------------------------
    def add_episode(self, episode: Episode) -> Episode:
        self._exec("INSERT OR REPLACE INTO episodes VALUES (?,?,?)",
                   (episode.user_id, episode.id, json.dumps(episode.to_dict())))
        return episode

    def get_episode(self, user_id: str, episode_id: str) -> Episode | None:
        r = self._rows("SELECT data FROM episodes WHERE id=? AND user_id=?",
                       (episode_id, user_id))
        return Episode.from_dict(json.loads(r[0]["data"])) if r else None

    def episodes(self, user_id: str) -> list[Episode]:
        return [Episode.from_dict(json.loads(r["data"])) for r in
                self._rows("SELECT data FROM episodes WHERE user_id=?", (user_id,))]

    def delete_episode(self, user_id: str, episode_id: str) -> bool:
        return self._exec("DELETE FROM episodes WHERE id=? AND user_id=?",
                          (episode_id, user_id)).rowcount > 0

    # --- events -----------------------------------------------------------
    def _put_event(self, e: Event) -> None:
        self._exec(
            "INSERT OR REPLACE INTO events VALUES (?,?,?,?,?,?,?)",
            (e.user_id, e.id, e.entity_id, e.branch, int(e.projected),
             e.order_key, json.dumps(e.to_dict())))

    def append_event(self, event: Event) -> Event:
        self._put_event(event)
        return event

    def get_event(self, user_id: str, event_id: str) -> Event | None:
        r = self._rows("SELECT data FROM events WHERE id=? AND user_id=?", (event_id, user_id))
        return Event.from_dict(json.loads(r[0]["data"])) if r else None

    def update_event(self, event: Event) -> Event | None:
        if not self._rows("SELECT 1 FROM events WHERE id=? AND user_id=?",
                          (event.id, event.user_id)):
            return None
        self._put_event(event)
        return event

    def delete_event(self, user_id: str, event_id: str) -> bool:
        return self._exec("DELETE FROM events WHERE id=? AND user_id=?",
                          (event_id, user_id)).rowcount > 0

    def events(self, user_id: str, *, entity_id: str | None = None,
               branch: str | None = None, since: str | None = None,
               until: str | None = None) -> list[Event]:
        sql = "SELECT data FROM events WHERE user_id=?"
        params: list = [user_id]
        if entity_id is not None:
            sql += " AND entity_id=?"; params.append(entity_id)
        if branch is not None:
            sql += " AND branch=?"; params.append(branch)
        if since is not None:
            sql += " AND order_key>=?"; params.append(since)
        if until is not None:
            sql += " AND order_key<=?"; params.append(until)
        sql += " ORDER BY order_key"
        return [Event.from_dict(json.loads(r["data"])) for r in self._rows(sql, params)]

    def unprojected_events(self, user_id: str) -> list[Event]:
        return [Event.from_dict(json.loads(r["data"])) for r in
                self._rows("SELECT data FROM events WHERE user_id=? AND projected=0", (user_id,))]

    # --- entity registry --------------------------------------------------
    def add_entity(self, entity: EntityRecord) -> EntityRecord:
        self._exec("INSERT OR REPLACE INTO entities VALUES (?,?,?)",
                   (entity.user_id, entity.entity_id, json.dumps(entity.to_dict())))
        return entity

    def get_entity(self, user_id: str, entity_id: str) -> EntityRecord | None:
        r = self._rows("SELECT data FROM entities WHERE entity_id=? AND user_id=?",
                       (entity_id, user_id))
        return EntityRecord.from_dict(json.loads(r[0]["data"])) if r else None

    def entity_by_alias(self, user_id: str, name: str) -> EntityRecord | None:
        needle = name.casefold().strip()
        for e in self.entities(user_id):
            if any(a.casefold().strip() == needle for a in e.aliases):
                return e
        return None

    def entities(self, user_id: str) -> list[EntityRecord]:
        return [EntityRecord.from_dict(json.loads(r["data"])) for r in
                self._rows("SELECT data FROM entities WHERE user_id=?", (user_id,))]

    def update_entity(self, entity: EntityRecord) -> EntityRecord | None:
        if not self._rows("SELECT 1 FROM entities WHERE entity_id=? AND user_id=?",
                          (entity.entity_id, entity.user_id)):
            return None
        self.add_entity(entity)
        return entity

    def delete_entity(self, user_id: str, entity_id: str) -> bool:
        return self._exec("DELETE FROM entities WHERE entity_id=? AND user_id=?",
                          (entity_id, user_id)).rowcount > 0

    # --- branch registry --------------------------------------------------
    def add_branch(self, branch: BranchRecord) -> BranchRecord:
        self._exec("INSERT OR REPLACE INTO branches VALUES (?,?,?,?)",
                   (branch.user_id, branch.entity_id, branch.branch,
                    json.dumps(branch.to_dict())))
        return branch

    def get_branch(self, user_id: str, entity_id: str, branch: str) -> BranchRecord | None:
        r = self._rows("SELECT data FROM branches WHERE user_id=? AND entity_id=? AND branch=?",
                       (user_id, entity_id, branch))
        return BranchRecord.from_dict(json.loads(r[0]["data"])) if r else None

    def branches(self, user_id: str, entity_id: str | None = None) -> list[BranchRecord]:
        if entity_id is None:
            rows = self._rows("SELECT data FROM branches WHERE user_id=?", (user_id,))
        else:
            rows = self._rows("SELECT data FROM branches WHERE user_id=? AND entity_id=?",
                              (user_id, entity_id))
        return [BranchRecord.from_dict(json.loads(r["data"])) for r in rows]

    def update_branch(self, branch: BranchRecord) -> BranchRecord | None:
        if self.get_branch(branch.user_id, branch.entity_id, branch.branch) is None:
            return None
        self.add_branch(branch)
        return branch

    def delete_branch(self, user_id: str, entity_id: str, branch: str) -> bool:
        return self._exec(
            "DELETE FROM branches WHERE user_id=? AND entity_id=? AND branch=?",
            (user_id, entity_id, branch)).rowcount > 0

    # --- maintenance ------------------------------------------------------
    def clear(self, user_id: str) -> bool:
        existed = False
        for table in ("episodes", "events", "entities", "branches"):
            if self._exec(f"DELETE FROM {table} WHERE user_id=?", (user_id,)).rowcount > 0:
                existed = True
        return existed
