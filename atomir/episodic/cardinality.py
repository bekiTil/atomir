"""Branch cardinality classifier for `BranchRecord.cardinality`.

- "single": new fact demotes (or overwrites) the previous fact on the branch.
- "multi": every write keeps previous facts live.
- "set": like "multi" but skips duplicate values.

`classify_cardinality` runs once per new branch canonical and caches the
result in-process. Env overrides skip the LLM.
"""
from __future__ import annotations

import os

_LLM_CACHE: dict[str, str] = {}


def _env_multi() -> set[str]:
    raw = os.environ.get("ATOMIR_MULTI_BRANCHES", "")
    return {t.strip().lower() for t in raw.split(",") if t.strip()}


def _env_single() -> set[str]:
    raw = os.environ.get("ATOMIR_SINGLE_BRANCHES", "")
    return {t.strip().lower() for t in raw.split(",") if t.strip()}


_CLASSIFY_SYSTEM = (
    "Classify a predicate as SINGLE- or MULTI-valued for a personal-memory "
    "store deciding whether a new fact on that predicate should replace the "
    "previous one or coexist with it.\n"
    "- SINGLE: only ONE simultaneously-true value at a time (employer, "
    "current address, spouse, birthplace).\n"
    "- MULTI: many simultaneously-true values (hobbies, places visited, "
    "books read, languages spoken, activities, conditions).\n"
    "When in doubt, prefer MULTI.\n"
    'Reply ONLY with JSON: {"cardinality": "single"} or {"cardinality": "multi"}.'
)


def classify_cardinality(llm, canonical: str, description: str = "",
                          object_type: str = "") -> str:
    """Return "single" or "multi" for a branch canonical. Cached per-process."""
    if not canonical:
        return "single"
    key = canonical.strip().lower()
    if key in _env_single():
        return "single"
    if key in _env_multi():
        return "multi"
    if key in _LLM_CACHE:
        return _LLM_CACHE[key]
    if llm is None:
        return "single"
    user = (f"predicate: {canonical}\n"
            f"description: {description or '(none)'}\n"
            f"object_type: {object_type or '(none)'}")
    try:
        res = llm.chat_json(_CLASSIFY_SYSTEM, user)
        val = ""
        if isinstance(res, dict):
            val = str(res.get("cardinality", "")).strip().lower()
        card = "multi" if val == "multi" else "single"
    except Exception:
        card = "single"
    _LLM_CACHE[key] = card
    return card
