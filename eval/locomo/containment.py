"""Containment instrument for the episodic read path.

It measures EPISODE-LEVEL CONTAINMENT: for each question, did the retriever pull an
event that came from the question's gold evidence message into the top-k? This is a
retrieval metric (no answerer), so it is free and deterministic given fixed embeddings.

Method (the reusable core):
  LoCoMo `dia_id`  --exact text join-->  stored Episode  --episode_id-->  Events
  A question's gold events are the events whose episode_id matches its evidence turns.
  "found@k" = any gold event is in the retriever's top-k.

The join is on episode TEXT ("Speaker: text", with a startswith fallback because
image-caption turns append a suffix on ingest), NOT on episode_id — ids regenerate
per ingest and would never match across stores.

This module is intentionally dependency-light and parameterized: point it at a store
JSON, a LoCoMo dataset, and a conversation index, and pass two ranker callables
(baseline vs candidate) that map (question_text, events) -> ranked list of event ids.
Report found@k and the paired discordant counts (b, c): b = questions the baseline
finds but the candidate does not, c = the reverse.
"""
from __future__ import annotations

import collections
import re

_DATE = re.compile(
    r"(?i)\b(yesterday|today|tomorrow|last|next|ago|this (week|month|year|morning|weekend)|"
    r"week|month|year|since|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|monday|tuesday|"
    r"wednesday|thursday|friday|saturday|sunday|\d{4}|\d{1,2}(st|nd|rd|th))\b")


def dia_to_episode(conversation: dict, episodes: list[dict]) -> dict[str, str]:
    """Map each LoCoMo dia_id to a stored episode id by exact/startswith text join."""
    turns = {}
    for v in conversation.values():
        if isinstance(v, list):
            for t in v:
                if isinstance(t, dict) and "dia_id" in t:
                    turns[t["dia_id"]] = f"{t['speaker']}: {t['text']}"
    out = {}
    for dia, key in turns.items():
        hit = [e["id"] for e in episodes if e["text"] == key] \
            or [e["id"] for e in episodes if e["text"].startswith(key)]
        out[dia] = hit[0] if len(hit) == 1 else None
    return out


def gold_events(qa: dict, ep_of: dict[str, str],
                events_by_episode: dict[str, set]) -> set:
    g = set()
    for dia in (qa.get("evidence") or []):
        ep = ep_of.get(dia)
        if ep:
            g |= events_by_episode.get(ep, set())
    return g


def temporal_questions(qa_list: list[dict], conversation: dict) -> list[dict]:
    """Canonical rule: category-2 (temporal) questions whose gold evidence turn text
    carries a date signal (so the question is answerable from its evidence)."""
    turn_text = {}
    for v in conversation.values():
        if isinstance(v, list):
            for t in v:
                if isinstance(t, dict) and "dia_id" in t:
                    turn_text[t["dia_id"]] = t.get("text", "")
    out = []
    for q in qa_list:
        if q.get("category") != 2:
            continue
        gt = " ".join(turn_text.get(d, "") for d in (q.get("evidence") or []))
        if _DATE.search(gt):
            out.append(q)
    return out


def gate(qas, ep_of, events_by_episode, baseline_ranker, candidate_ranker, ks=(10, 20, 50)):
    """baseline_ranker / candidate_ranker: (question_text) -> ranked list of event ids.
    Returns {k: (baseline_found, candidate_found, b, c, n)}."""
    out = {}
    per_q = []
    for q in qas:
        g = gold_events(q, ep_of, events_by_episode)
        if not g:
            continue
        per_q.append((g, baseline_ranker(q["question"]), candidate_ranker(q["question"])))
    for k in ks:
        bf = cf = b = c = 0
        for g, base, cand in per_q:
            fb = bool(g & set(base[:k])); fc = bool(g & set(cand[:k]))
            bf += fb; cf += fc; b += (fb and not fc); c += ((not fb) and fc)
        out[k] = (bf, cf, b, c, len(per_q))
    return out
