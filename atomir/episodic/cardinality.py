"""Branch cardinality classifier — turns on the `multi`/`set` code paths that
`atomir.episodic.projection.project_event` already supports.

The projection layer honours `BranchRecord.cardinality`:
- "single" (default) — a new fact demotes (append_only) or overwrites the
  previous fact on the branch. Correct for one-at-a-time relations like
  `works_at`, `lives_in`, `is_married_to`.
- "multi" — every write keeps the previous facts live; nothing is demoted.
  Correct for many-at-a-time relations like `engaged_in`, `visited`,
  `has_hobby`, `speaks`, `mentioned`.
- "set" — like `multi` but skips duplicates by normalised value.

Classification is per-branch and permanent (stored on `BranchRecord`). The
classifier below asks the LLM once when a NEW branch is being created and
caches the answer in-process so the same canonical isn't classified twice
per session. An env override lets you force additions without an LLM call.
"""
from __future__ import annotations

import os

# In-process cache: canonical name (lowercased) -> "single" | "multi".
# Keeps repeat classifications free within one session. Callers may reset it
# in tests by clearing this dict directly.
_LLM_CACHE: dict[str, str] = {}


def _env_multi() -> set[str]:
    """Force-additions via ATOMIR_MULTI_BRANCHES (comma-separated canonicals).
    Applied BEFORE the LLM call — a match skips the LLM entirely."""
    raw = os.environ.get("ATOMIR_MULTI_BRANCHES", "")
    return {t.strip().lower() for t in raw.split(",") if t.strip()}


def _env_single() -> set[str]:
    """Force-additions via ATOMIR_SINGLE_BRANCHES — same as _env_multi but
    forces `single`. Useful to override LLM misclassifications per-deploy."""
    raw = os.environ.get("ATOMIR_SINGLE_BRANCHES", "")
    return {t.strip().lower() for t in raw.split(",") if t.strip()}


_CLASSIFY_SYSTEM = (
    "You classify a personal-memory predicate as SINGLE-valued or MULTI-"
    "valued so a memory store knows whether a new fact on that predicate "
    "should REPLACE the previous one or coexist with it.\n"
    "- SINGLE: a person naturally has only ONE simultaneously-true value on "
    "this predicate at any given time — a NEW value invalidates the OLD. "
    "Examples: employer (primary job), current address, spouse, birthplace, "
    "primary employer, current job title.\n"
    "- MULTI: a person naturally has MANY simultaneously-true values on this "
    "predicate — the old values stay true when new ones arrive. Examples: "
    "hobbies, places visited, books read, languages spoken, activities "
    "engaged in, things mentioned, foods eaten, condition (a person can have "
    "several), owns.\n"
    "When in doubt, prefer MULTI (over-preserving history is safer than "
    "wrongly discarding it).\n"
    'Reply ONLY with JSON: {"cardinality": "single"} or {"cardinality": "multi"}.'
)


def classify_cardinality(llm, canonical: str, description: str = "",
                          object_type: str = "") -> str:
    """Return 'single' or 'multi' for a branch canonical. Cached per-process
    by canonical name. `llm` may be None — in that case defaults to 'single'."""
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
