"""Single-extraction: one LLM call turns a message into structured events
. This REPLACES the fact-extraction call when episodic is on —
facts are then projected from these events, never extracted independently
("single extraction, dual projection", spec).
"""

from __future__ import annotations

from atomir.episodic.models import HAPPENED, START
from atomir.ontology import normalize_object_type
from atomir.providers.llm_base import LLM

import os as _os
import json as _json
import hashlib as _hashlib

# Opt-in content-hash cache (ATOMIR_EXTRACT_CACHE). Keys extraction on the message
# content (prompt_version, message_date, message_text), so re-ingesting the same
# conversation returns identical extractions. Default OFF. Set to a file path (or
# "1" for a default path) to enable.
_PROMPT_VERSION = None
_CACHE = None
def _cache_path():
    v = _os.environ.get("ATOMIR_EXTRACT_CACHE")
    if not v:
        return None
    return "./atomir_extract_cache.json" if v == "1" else v
def _cache_load():
    global _CACHE
    if _CACHE is None:
        _CACHE = {}
        p = _cache_path()
        if p and _os.path.exists(p):
            try:
                _CACHE = _json.load(open(p))
            except Exception:
                _CACHE = {}
    return _CACHE
def _cache_key(text, recorded_at):
    global _PROMPT_VERSION
    if _PROMPT_VERSION is None:
        _PROMPT_VERSION = _hashlib.sha1(_EXTRACT_SYSTEM.encode("utf-8")).hexdigest()[:8]
    raw = f"{_PROMPT_VERSION}\x00{recorded_at}\x00{text}"
    return _hashlib.sha1(raw.encode("utf-8")).hexdigest()

_EXTRACT_SYSTEM = (
    "You extract EVENTS (things that happened or were stated) from a message for a "
    "personal-memory system. The message was written by the user, so the default "
    "subject is 'the user'. Return ONE item per atomic event.\n"
    "For each event give:\n"
    "- raw_text: the source clause, verbatim\n"
    "- subject: who the event is about ('the user' unless clearly someone else)\n"
    "- subject_type: person | org | ...\n"
    "- verb_phrase: the predicate WITHOUT the object, e.g. 'joined', 'left', "
    "'reports to', 'lives in'\n"
    "- value: the object/value ONLY, e.g. 'Acme Corp', 'Dana', 'Paris', 'friends'. "
    "Do NOT put a duration or age here.\n"
    "- qualifier: a time-QUANTITY that modifies the value — how long, how old, or "
    "since when ('4 years', '28 years old', 'since 2019', '10 years ago'); null if "
    "none. Keep the object in `value` and the span in `qualifier`: 'I've known "
    "these friends for 4 years' -> value='friends', qualifier='4 years'.\n"
    "- object_type: choose ONE of: organization | person | place | activity | "
    "object | language | media | animal | condition | other\n"
    "- polarity: 'start' (relationship/state begins), 'end' (it ceases), or "
    "'update' (same relationship, new value)\n"
    "- modality: 'happened' (real, past/present), 'planned' (future/intended), or "
    "'hypothetical' (uncertain/conditional)\n"
    "- occurred_at: ISO-8601 date the event happened in the world, resolving "
    "relative times ('last November') against the message date; null if unknown\n"
    "- corrects_hint: brief text of an earlier statement this corrects, or null\n"
    "Split compound statements. Skip greetings and chit-chat.\n"
    'Return ONLY JSON: {"events": [ { ...fields... } ]}'
)


def extract_events(llm: LLM, text: str, recorded_at: str, registry: str = "") -> list[dict]:
    """Return a list of raw event dicts. `recorded_at` (message time) lets the
    model resolve relative times into `occurred_at`; `registry` (the user's
    existing branches) is fed back so it reuses established predicates/verbs
    rather than inventing variants."""
    text = (text or "").strip()
    if not text:
        return []
    system = _EXTRACT_SYSTEM
    if registry:
        system += ("\n\nThe user already has these memory predicates — REUSE an "
                   "existing verb_phrase/object_type when the event fits one, "
                   "rather than phrasing it differently:\n" + registry)
    user = f"MESSAGE DATE: {recorded_at}\nMESSAGE:\n{text}"
    _ck = None
    if _cache_path() is not None:
        _cache = _cache_load()
        _ck = _cache_key(text, recorded_at)
        if _ck in _cache:
            return _cache[_ck]  # reproducible: extraction served from cache
    result = llm.chat_json(system, user)
    events = result.get("events", []) if isinstance(result, dict) else []
    out: list[dict] = []
    for e in events:
        if not isinstance(e, dict):
            continue
        verb = str(e.get("verb_phrase", "")).strip()
        value = str(e.get("value", "")).strip()
        if not verb and not value:
            continue
        out.append({
            "raw_text": str(e.get("raw_text", "") or text).strip(),
            "subject": str(e.get("subject", "the user")).strip() or "the user",
            "subject_type": str(e.get("subject_type", "person")).strip() or "person",
            "verb_phrase": verb,
            "value": value,
            "object_type": normalize_object_type(str(e.get("object_type", ""))),
            "qualifier": (str(e["qualifier"]).strip() if e.get("qualifier") else None),
            "polarity": _norm_polarity(e.get("polarity")),
            "modality": _norm_modality(e.get("modality")),
            "occurred_at": _norm_ts(e.get("occurred_at")),
            "corrects_hint": (str(e["corrects_hint"]).strip()
                              if e.get("corrects_hint") else None),
        })
    if _ck is not None:
        _cache = _cache_load()
        _cache[_ck] = out
        p = _cache_path()
        if p:
            _json.dump(_cache, open(p, "w"))
    return out


def _norm_polarity(v) -> str:
    v = str(v or "").strip().lower()
    return v if v in {"start", "end", "update"} else START


def _norm_modality(v) -> str:
    v = str(v or "").strip().lower()
    return v if v in {"happened", "planned", "hypothetical"} else HAPPENED


def _norm_ts(v):
    v = (str(v).strip() if v is not None else "")
    return v or None
