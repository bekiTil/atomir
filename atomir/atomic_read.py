"""Atomic READ: answer a question by querying memory atomically.

The read-side twin of the write engine. Instead of embedding a whole question
as one blob, a planner may decompose it into atomic sub-questions; each is
retrieved independently and the results are unioned (deduped by fact id, best
score kept). Vendor-neutral: imports ONLY the injected interfaces.
"""

from __future__ import annotations

import os
import re
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from atomir.providers.embedder_base import Embedder
from atomir.providers.llm_base import LLM
from atomir.ranking import BM25, rrf
from atomir.store_base import MemoryStore

_PLAN_SYSTEM = (
    "You are a query planner for a personal-memory system. Decide whether a "
    "question must be broken into atomic sub-questions to answer it.\n"
    "- SIMPLE single-fact question -> decompose=false and subquestions=[the "
    "original question verbatim].\n"
    "- MULTI-HOP question (the answer depends on an intermediate fact, e.g. a "
    "person, project, or relationship you must resolve first) -> decompose=true "
    "with atomic wh- sub-questions, INCLUDING the intermediate facts.\n\n"
    "Write every sub-question in the THIRD PERSON about \"the user\" (facts are "
    'stored that way, e.g. "what ingredients does the user have?", never "what '
    'ingredients do I have?"). Keep them short and in the vocabulary a stored '
    'fact would use (prefer "who is the user\'s manager?" over "can you tell me '
    'about the person who manages the user?").\n\n'
    "Examples:\n"
    'Q: "where does the user live?"\n'
    '{"decompose": false, "subquestions": ["where does the user live?"]}\n'
    'Q: "what is the user\'s favorite food?"\n'
    '{"decompose": false, "subquestions": ["what is the user\'s favorite food?"]}\n'
    'Q: "what gift should I buy my sister?"\n'
    '{"decompose": true, "subquestions": ["who is the user\'s sister?", '
    '"what does the user\'s sister like?"]}\n'
    'Q: "who should I email about my project?"\n'
    '{"decompose": true, "subquestions": ["what project is the user working on?", '
    '"who leads the user\'s team?"]}\n'
    'Q: "when did the user start at their current job?"\n'
    '{"decompose": true, "subquestions": ["where does the user work?", '
    '"when did the user start that job?"]}\n'
    'Q: "can my manager and I both make the Friday meeting?"\n'
    '{"decompose": true, "subquestions": ["who is the user\'s manager?", '
    '"when is the user available?", "when is the manager available?"]}\n\n'
    "Respond ONLY with JSON: "
    '{"decompose": <true|false>, "subquestions": ["...", "..."]}'
)


_FILLER = re.compile(
    r"\b(please|kindly|can you|could you|would you|tell me|"
    r"i want to know|i would like to know|do you know)\b"
)


def canonical_query(query: str) -> str:
    """Normal form used as the plan cache key: lowercased, filler and
    punctuation stripped, whitespace collapsed. Deliberately conservative so
    two genuinely different questions never collide onto the same key."""
    q = _FILLER.sub(" ", query.lower())
    q = re.sub(r"[^\w\s]", " ", q)
    return re.sub(r"\s+", " ", q).strip()


def _canonical_subqs(subs: list[str]) -> list[str]:
    """Trim, collapse whitespace, and drop case-insensitive duplicates."""
    seen: set[str] = set()
    out: list[str] = []
    for s in subs:
        s = re.sub(r"\s+", " ", s.strip())
        key = s.lower()
        if s and key not in seen:
            seen.add(key)
            out.append(s)
    return out


class PlanCache:
    """Bounded LRU of canonical_query -> plan. Decomposition depends only on the
    query text (not the user or the store), so entries are shared safely."""

    def __init__(self, maxsize: int = 512) -> None:
        self.maxsize = maxsize
        self._d: "OrderedDict[str, dict]" = OrderedDict()

    def get(self, key: str) -> dict | None:
        if key in self._d:
            self._d.move_to_end(key)
            return self._d[key]
        return None

    def put(self, key: str, value: dict) -> None:
        self._d[key] = value
        self._d.move_to_end(key)
        while len(self._d) > self.maxsize:
            self._d.popitem(last=False)

    def clear(self) -> None:
        self._d.clear()


_PLAN_CACHE = PlanCache()


def _copy_plan(p: dict) -> dict:
    # Return a copy so callers can never mutate a cached entry in place.
    return {"decompose": p["decompose"], "subquestions": list(p["subquestions"])}


def _plan_llm(llm: LLM, query: str) -> dict:
    result = llm.chat_json(_PLAN_SYSTEM, query)
    decompose = bool(result.get("decompose", False))
    subs = _canonical_subqs(
        [s for s in (result.get("subquestions") or []) if isinstance(s, str)]
    )
    if not subs:  # planner gave nothing usable -> fall back to the raw query
        subs, decompose = [query], False
    return {"decompose": decompose, "subquestions": subs}


def plan(llm: LLM, query: str, *, cache: bool = True) -> dict:
    """Decide whether/how to decompose `query`. Always returns >=1 subquestion.

    With `cache` (default), the plan is memoized by canonical query form, so a
    repeated query skips the planner LLM call entirely — 0ms and deterministic.
    """
    key = canonical_query(query) if cache else ""
    if key:
        hit = _PLAN_CACHE.get(key)
        if hit is not None:
            return _copy_plan(hit)
    result = _plan_llm(llm, query)
    if key:
        _PLAN_CACHE.put(key, _copy_plan(result))
    return result


def atomic_search(
    store: MemoryStore,
    llm: LLM,
    embedder: Embedder,
    user_id: str,
    query: str,
    k: int = 6,
    decompose: bool = True,
    hybrid: bool = True,
    corpus_limit: int = 5000,
    cache_plans: bool = True,
    only_current: bool = True,
    episodic=None,
) -> dict:
    """Retrieve facts for `query`, decomposing into sub-questions when useful.

    With `hybrid` (default), each sub-question contributes TWO rankings — dense
    (embedding similarity) and lexical (BM25 over the user's facts) — and all of
    them are fused with Reciprocal Rank Fusion. Semantic search alone buries
    facts that hinge on rare terms (names, project titles); BM25 recovers them.

    Returns {"subquestions": [...], "results": [...]} — sub-questions are
    exposed deliberately (the differentiator, and a debugging aid). With hybrid,
    `score` is the fused RRF score (rank-based), not a cosine.
    """
    # Overlap the raw-query embed with the planner call (independent); reuse it
    # when the query isn't decomposed.
    prefetched: dict[str, list[float]] = {}
    if decompose:
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_plan = ex.submit(plan, llm, query, cache=cache_plans)
            fut_emb = ex.submit(embedder.embed_query, query)
            subquestions = fut_plan.result()["subquestions"]
            try:
                prefetched[query] = fut_emb.result()
            except Exception:  # prefetch is speculative; never fail the search on it
                pass
    else:
        subquestions = [query]

    # Pool per ranking: wider than k so fusion has candidates to promote.
    pool = max(k * 5, 50)
    by_id: dict[str, dict] = {}

    # Lexical index over the user's facts, built once and scored per sub-question.
    # When ATOMIR_BM25_INCLUDE_RAWTEXT is on and an episodic store is available,
    # a SECOND BM25 index is built over each fact's per-event raw_text and fused
    # into RRF as an extra rank list. Rare qualifier words the projection drops
    # (adjectives, brand names, exact phrases) become matchable without diluting
    # the primary fact-BM25 signal — the two indexes stay separate and both
    # contribute rank votes. The dense embedding side is unchanged.
    bm25 = None
    bm25_raw = None
    corpus_ids: list[str] = []
    if hybrid:
        corpus = store.all(user_id, limit=corpus_limit)
        if corpus:
            corpus_ids = [f["id"] for f in corpus]
            bm25 = BM25([f["text"] for f in corpus])
            include_raw = (
                episodic is not None
                and os.environ.get("ATOMIR_BM25_INCLUDE_RAWTEXT", "off") == "on"
            )
            if include_raw:
                raw_by_fact: dict[str, list[str]] = {}
                try:
                    for e in episodic.events(user_id):
                        fid = getattr(e, "fact_id", None)
                        if not fid:
                            continue
                        # Combine raw_text with the extractor's verb_raw so
                        # the specific verb ("reading", "hiked") is BM25-
                        # matchable even though branch normalisation folded it
                        # into a canonical predicate.
                        rt = (getattr(e, "raw_text", "") or "").strip()
                        vr = (getattr(e, "verb_raw", "") or "").strip()
                        combined = " ".join(x for x in (rt, vr) if x)
                        if combined:
                            raw_by_fact.setdefault(fid, []).append(combined)
                except Exception:
                    raw_by_fact = {}
                raw_docs = [" ".join(raw_by_fact.get(f["id"], [])) for f in corpus]
                if any(raw_docs):
                    bm25_raw = BM25(raw_docs)
            by_id.update({f["id"]: f for f in corpus})

    # Independent per sub-question -> retrieve concurrently; fusion is order-free.
    def _dense(sq: str) -> list[dict]:
        vec = prefetched.get(sq) or embedder.embed_query(sq)
        return store.search(user_id, vec, k=pool)

    if len(subquestions) == 1:
        dense_hits = [_dense(subquestions[0])]
    else:
        with ThreadPoolExecutor(max_workers=min(len(subquestions), 8)) as ex:
            dense_hits = list(ex.map(_dense, subquestions))

    rankings: list[list[str]] = []
    for sq, hits in zip(subquestions, dense_hits):
        for h in hits:
            by_id.setdefault(h["id"], h)
        rankings.append([h["id"] for h in hits])  # dense ranking
        if bm25 is not None:
            scored = [(i, s) for i, s in enumerate(bm25.scores(sq)) if s > 0]
            scored.sort(key=lambda t: t[1], reverse=True)
            rankings.append([corpus_ids[i] for i, _ in scored[:pool]])  # lexical
        if bm25_raw is not None:
            # Third RRF signal: BM25 over per-fact raw_text (the actual clauses
            # the user said, kept verbatim on each event). Fuses in as a rank
            # list — every candidate this ranker surfaces adds a small vote to
            # RRF without altering the fact-BM25 or dense rankings.
            scored_raw = [(i, s) for i, s in enumerate(bm25_raw.scores(sq)) if s > 0]
            scored_raw.sort(key=lambda t: t[1], reverse=True)
            rankings.append([corpus_ids[i] for i, _ in scored_raw[:pool]])  # raw-text

    def _live(f: dict) -> bool:
        """Under append_only_facts mode, facts marked is_current=False are
        historical. When only_current is True, filter them out. When False,
        pass them through (semantic route sees history)."""
        if not only_current:
            return True
        meta = f.get("metadata") or {}
        return meta.get("is_current") is not False  # missing key = live (0.8.4)

    if not hybrid:  # dense-only: rank by raw similarity, as before
        best: dict[str, dict] = {}
        for hits in dense_hits:
            for h in hits:
                if h["id"] not in best or h["score"] > best[h["id"]]["score"]:
                    best[h["id"]] = h
        results = sorted(best.values(), key=lambda h: h["score"], reverse=True)
        results = [r for r in results if _live(r)][:k]
        return {"subquestions": subquestions, "results": results}

    fused = rrf(rankings)
    top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    results = [{**by_id[fid], "score": score} for fid, score in top
               if fid in by_id and _live(by_id[fid])][:k]
    return {"subquestions": subquestions, "results": results}
