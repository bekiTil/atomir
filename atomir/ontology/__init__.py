"""Ontology support for the episodic layer.

atomir is general-purpose: by default a user starts with an EMPTY branch registry
and the ontology emerges from their own messages (kept consistent by feeding the
existing registry back into the extractor/namer/planner prompts — see
`format_registry`). Optional, swappable *packs* pre-seed a registry for a known
domain; `personal` ships as an example.

Pack format: a list of tuples
    (canonical, state_template, object_type, [aliases...], description)
`object_type` must be one of OBJECT_TYPES. Register a pack by adding it to PACKS
(or pass your own list to `EpisodicMemory(ontology_pack=...)` indirectly via
`apply_pack`). Select a shipped pack with the ONTOLOGY_PACK env var (default "" =
none).
"""

from __future__ import annotations

from atomir.episodic.models import BranchRecord

# Fixed object-type enum — the extractor and namer choose from these.
OBJECT_TYPES = [
    "organization", "person", "place", "activity", "object",
    "language", "media", "animal", "condition", "other",
]

_SYNONYMS = {
    "org": "organization", "company": "organization", "employer": "organization",
    "city": "place", "town": "place", "country": "place", "location": "place",
    "hobby": "activity", "sport": "activity", "instrument": "object",
    "vehicle": "object", "car": "object", "tool": "object", "equipment": "object",
    "food": "object", "book": "media", "publication": "media", "pet": "animal",
    "dog": "animal", "cat": "animal", "illness": "condition", "disease": "condition",
    "goal": "other",
}


def normalize_object_type(raw: str) -> str:
    """Map a free-text object type onto the enum (prefix-tolerant), else 'other'."""
    r = (raw or "").lower().strip()
    if not r:
        return "other"
    if r in OBJECT_TYPES:
        return r
    if r in _SYNONYMS:
        return _SYNONYMS[r]
    for t in OBJECT_TYPES:
        if r.startswith(t) or t.startswith(r):
            return t
    return "other"


def format_registry(branches, cap: int = 40) -> str:
    """Render a user's existing branches as a reuse-first vocabulary for prompts
    (canonical: description [object_type]). Capped so prompts stay bounded.

    Sorted by branch name so the prompt is DETERMINISTIC: without a stable order,
    which entries survive the `cap` (and in what order) can vary between runs.
    Sorting is language-level (no dataset dependence)."""
    ordered = sorted(branches, key=lambda b: b.branch)
    lines = [f"- {b.branch}: {b.description} [{b.object_type or 'other'}]"
             for b in ordered[:cap]]
    return "\n".join(lines)


def branch_vocab(branches) -> list[str]:
    """The user's actual branch canonicals — the planner's branch_hint vocabulary."""
    return [b.branch for b in branches]


def apply_pack(store, user_id: str, entity_id: str, seeds) -> int:
    """Copy a pack's branches into a user's registry under `entity_id`. Idempotent
    (skips canonicals that already exist). Returns how many were added."""
    have = {b.branch for b in store.branches(user_id, entity_id)}
    added = 0
    for canonical, template, obj_type, aliases, desc in seeds:
        if canonical in have:
            continue
        # Seeded branches keep the default cardinality; override via
        # ATOMIR_MULTI_BRANCHES if needed.
        store.add_branch(BranchRecord(
            branch=canonical, user_id=user_id, entity_id=entity_id,
            description=desc, state_template=template, object_type=obj_type,
            aliases=list(aliases)))
        added += 1
    return added


def get_pack(name: str):
    """Return a shipped pack's seeds by name, or None if unknown/empty."""
    if not name:
        return None
    from atomir.ontology import personal
    return {"personal": personal.PERSONAL_SEEDS}.get(name.strip().lower())
