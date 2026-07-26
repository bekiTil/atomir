"""Episodic data models.

Plain dataclasses so any store can persist them. Timestamps are ISO-8601
strings (UTC) rather than datetimes: JSON round-trips trivially and ISO strings
sort chronologically, which is all the chain walk needs. `occurred_at` may be
None (unknown event time); `recorded_at` is always set (message time).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from uuid import uuid4

# Polarity: what an event does to the branch's state.
START, END, UPDATE = "start", "end", "update"
# Modality: only "happened" events project to facts.
HAPPENED, PLANNED, HYPOTHETICAL = "happened", "planned", "hypothetical"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def _from_dict(cls, d: dict):
    """Rebuild a dataclass from a stored dict, ignoring unknown keys and letting
    missing keys fall back to defaults (so old records stay loadable)."""
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Event:
    id: str
    user_id: str
    entity_id: str              # SUBJECT entity (the user's self-entity by default)
    branch: str                 # canonical predicate, e.g. "works_at"
    value: str                  # object/value, e.g. "Acme Corp"
    raw_text: str               # the source clause as the user said it
    polarity: str               # START | END | UPDATE
    recorded_at: str            # ISO; when the user said it (always set)
    # --- optional (defaulted so old records stay loadable) ---
    value_entity_id: str | None = None  # object resolved to an entity, if any
    occurred_at: str | None = None   # ISO; when it happened (None if unknown)
    modality: str = HAPPENED    # HAPPENED | PLANNED | HYPOTHETICAL
    corrects: str | None = None      # event id this corrects (not a state change)
    episode_id: str = ""        # source Episode id
    projected: bool = False     # write-ahead flag
    fact_id: str | None = None  # the fact this event projected to
    kind: str = "event"         # "event" | "checkpoint"
    checkpointed: bool = False  # rolled into a checkpoint summary
    superseded: bool = False    # corrected by a later event (excluded from the chain)

    @property
    def order_key(self) -> str:
        """Chain ordering: by occurred_at, falling back to recorded_at."""
        return self.occurred_at or self.recorded_at

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Event":
        return _from_dict(cls, d)


@dataclass
class Episode:
    id: str
    user_id: str
    text: str                   # raw message, verbatim (lossless bottom layer)
    created_at: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Episode":
        return _from_dict(cls, d)


@dataclass
class EntityRecord:
    entity_id: str
    user_id: str
    canonical_name: str
    aliases: list[str] = field(default_factory=list)
    entity_type: str | None = None   # "person" | "org" | ... (informational)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EntityRecord":
        return _from_dict(cls, d)


@dataclass
class BranchRecord:
    branch: str                 # canonical predicate: "works_at"
    user_id: str
    entity_id: str              # branch is scoped per entity
    description: str            # human-readable, embedded for matching
    state_template: str         # e.g. "works at" -> "The user works at X"
    object_type: str = ""       # e.g. "organization" — merge guard (same-type only)
    aliases: list[str] = field(default_factory=list)
    # verbs recorded by value-anchored end resolution are QUARANTINED here,
    # not promoted to confirmed aliases — so "left" (anchored via "left Beta")
    # never makes "left the gym" exact-match employment.
    provisional_aliases: list[str] = field(default_factory=list)
    current_fact_id: str | None = None  # the branch's current projected fact
    current_value: str | None = None    # the current state's value (for end-matching)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BranchRecord":
        return _from_dict(cls, d)
