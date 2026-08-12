"""Episodic read path: typed sub-question decomposition + routing.

Each sub-question is typed and routed:
- temporal -> walk the entity/branch chain (deterministic, ordered by
  occurred_at; no embeddings). Answers "when did X change / what happened".
- current  -> fact retrieval (facts are the authoritative "now").
- semantic -> the existing hybrid dense+BM25 retrieval.
Results are deduplicated by the event<->fact link before returning.
"""

from __future__ import annotations

from atomir.atomic_read import atomic_search
from atomir.ranking import BM25
from atomir.episodic.models import Event, value_phrase
from atomir.episodic.store_base import EpisodicStore
from atomir.providers.embedder_base import Embedder
from atomir.providers.llm_base import LLM
from atomir.store_base import MemoryStore

TEMPORAL, CURRENT, SEMANTIC = "temporal", "current", "semantic"

_TYPED_PLAN_SYSTEM = (
    "You are a query planner for a personal-memory system that stores both facts "
    "(the current state) and a timeline of events. Decompose the question into "
    "atomic sub-questions and TYPE each one:\n"
    "- temporal: about WHEN something happened or how it changed over time "
    "('when did the user join', 'who did the user work for in December').\n"
    "- current: about the state right now ('who is the user's manager').\n"
    "- semantic: everything else / open-ended recall.\n"
    "entity_hint is the SUBJECT whose timeline to walk — usually 'the user', NOT "
    "the object being asked about ('which companies has the user worked for' -> "
    "entity_hint='the user', not 'companies'). branch_hint MUST be copied VERBATIM "
    "from this exact list of the user's branches, or be null if none fits — NEVER "
    "invent a name that is not in the list:\n{vocab}\n"
    "Examples:\n"
    'Q: "which companies has the user worked for?" -> {{"text": "which companies '
    'has the user worked for", "type": "temporal", "entity_hint": "the user", '
    '"branch_hint": "works_at"}}\n'
    'Q: "where does the user live now?" -> {{"text": "where does the user live", '
    '"type": "current", "entity_hint": "the user", "branch_hint": "lives_in"}}\n'
    'Q: "what are the user\'s hobbies?" -> {{"text": "what hobbies does the user '
    'have", "type": "semantic", "entity_hint": null, "branch_hint": null}}\n'
    "Open-ended preference/attribute questions (hobbies, likes, favorites, general "
    "recall) are SEMANTIC, not temporal. Write sub-questions in the third person "
    "about 'the user'.\n"
    'Return ONLY JSON: {{"subquestions": [ {{"text": "...", '
    '"type": "temporal|current|semantic", "entity_hint": null, "branch_hint": null}} ]}}'
)


def plan_typed(llm: LLM, query: str, vocab: list[str] | None = None) -> list[dict]:
    """Return a list of typed sub-questions; falls back to one semantic
    sub-question (the raw query) if the planner gives nothing usable. `vocab` is
    the user's actual branch canonicals — the branch_hint vocabulary."""
    vocab_str = ", ".join(vocab) if vocab else "(the user has no branches yet)"
    system = _TYPED_PLAN_SYSTEM.format(vocab=vocab_str)
    res = llm.chat_json(system, query)
    subs = res.get("subquestions") if isinstance(res, dict) else None
    out: list[dict] = []
    for s in subs or []:
        if not isinstance(s, dict) or not str(s.get("text", "")).strip():
            continue
        typ = str(s.get("type", SEMANTIC)).strip().lower()
        out.append({
            "text": str(s["text"]).strip(),
            "type": typ if typ in {TEMPORAL, CURRENT, SEMANTIC} else SEMANTIC,
            "entity_hint": (str(s["entity_hint"]).strip() if s.get("entity_hint") else None),
            "branch_hint": (str(s["branch_hint"]).strip() if s.get("branch_hint") else None),
        })
    return out or [{"text": query, "type": SEMANTIC, "entity_hint": None, "branch_hint": None}]


def _resolve_branch(episodic: EpisodicStore, user_id: str, entity_id: str,
                    branch_hint: str | None, embedder: Embedder | None = None,
                    floor: float = 0.30, margin: float = 0.10):
    """Resolve a planner branch_hint to a real branch.

    Step 1 — EXACT vocab lookup: a hint that names a real branch (canonical or
    alias) resolves immediately (a strong model that obeys the constraint short-
    circuits here). Step 2 — RELATIVE-BEST: weak models invent hints (works_at)
    that don't string-match emergent names (joined_organization), so score every
    branch and resolve to the top ONLY if it's a clear winner: top >= floor AND
    (top - second) >= margin. Otherwise unresolved -> the caller's bounded
    fallback. Relative separation, not an absolute threshold, is the right signal.
    """
    if not branch_hint:
        return None
    hint = branch_hint.strip().casefold()
    branches = episodic.branches(user_id, entity_id)
    for b in branches:  # step 1: exact vocab lookup
        if b.branch.casefold() == hint or hint in {a.casefold() for a in b.aliases} \
                or b.branch.replace("_", " ").casefold() == hint:
            return b
    if embedder is None or not branches:
        return None
    from atomir.episodic.registry import cosine
    hint_vec = embedder.embed_query(branch_hint.replace("_", " "))
    scored = []
    for b in branches:
        # All signal helps MATCHING (even quarantined aliases + the state
        # template), though only confirmed aliases drive exact matches.
        rep = " ".join([b.branch.replace("_", " "), b.state_template, b.description,
                        *b.aliases, *b.provisional_aliases])
        scored.append((cosine(hint_vec, embedder.embed_passage(rep)), b))
    scored.sort(key=lambda t: t[0], reverse=True)
    top_score, top = scored[0]
    second = scored[1][0] if len(scored) > 1 else 0.0
    if top_score >= floor and (top_score - second) >= margin:  # step 2: clear winner
        return top
    return None


def _adds_signal(raw: str, stem: str) -> bool:
    """True when the verbatim clause carries content words the (verb, value)
    projection dropped — so appending it isn't just an echo. General (based on
    novel word count), not tied to any question phrasing."""
    have = {w.strip('.,"()\'') for w in stem.lower().split()}
    novel = [w for w in raw.lower().split() if w.strip('.,"()\'') and w.strip('.,"()\'') not in have]
    return len(novel) >= 3


def _event_result(episodic: EpisodicStore, e: Event, append_raw: bool = False) -> dict:
    if e.kind == "checkpoint":
        text = f"(summary) {e.value}"
    else:
        verb = e.branch.replace("_", " ")
        neg = "no longer " if e.polarity == "end" else ""
        # Weave the qualifier back into the object ("friends (4 years)") so
        # duration/age questions are answerable — the span lives only here on the
        # read path, never in `value`.
        stem = f"The user {neg}{verb} {value_phrase(e.value, e.qualifier)}".strip()
        # Stamp the event's date into the retrievable text so "when did X happen"
        # questions can be answered — projection otherwise drops the date.
        date = (e.occurred_at or e.recorded_at or "")[:10]
        text = f"On {date}, {stem[0].lower()}{stem[1:]}" if date and stem else stem
        # Anti-loss: the (verb, value) projection routinely drops salient nouns
        # ("18th birthday" -> "hand-painted bowl"). Surface the source clause when
        # it carries content the projection lost — but ONLY for a direct temporal
        # lookup (append_raw). Multi-hop/open-domain queries compose several
        # retrievals where the extra text adds noise, so they keep the clean text.
        raw = (e.raw_text or "").strip()
        if append_raw and raw and _adds_signal(raw, stem):
            text = f'{text} ("{raw}")'
    return {
        "id": e.id, "type": TEMPORAL, "text": text, "kind": e.kind,
        "value": e.value, "polarity": e.polarity, "occurred_at": e.occurred_at,
        "recorded_at": e.recorded_at, "branch": e.branch, "fact_id": e.fact_id,
    }


def walk_chain(episodic: EpisodicStore, user_id: str, *, entity_id: str | None = None,
               branch: str | None = None, since: str | None = None,
               until: str | None = None) -> list[Event]:
    """Ordered chain for an entity/branch: checkpoint summaries + the
    non-checkpointed, non-superseded raw tail."""
    return [e for e in episodic.events(user_id, entity_id=entity_id, branch=branch,
                                       since=since, until=until)
            if not e.superseded and (e.kind == "checkpoint" or not e.checkpointed)]


def _expand_checkpoint_members(episodic: EpisodicStore, user_id: str,
                                pool: list[Event]) -> list[Event]:
    """For each checkpoint in `pool`, load its absorbed_event_ids and add
    those events to the pool for ranking. Deduplicates by event id.
    Checkpoints themselves stay in the pool. Members whose ids are missing
    (unbackfilled stores) or are superseded are skipped.
    """
    have = {e.id for e in pool}
    add: list[Event] = []
    for e in pool:
        if e.kind != "checkpoint" or not getattr(e, "absorbed_event_ids", None):
            continue
        for mid in e.absorbed_event_ids:
            if mid in have:
                continue
            m = episodic.get_event(user_id, mid)
            if m is None or m.superseded:
                continue
            add.append(m)
            have.add(mid)
    return pool + add


_MONTHS = ["", "January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"]


def _lex_text(e: Event) -> str:
    """Lexical field for BM25 on the temporal route: the returned event text plus the
    verbatim source clause and the date rendered in prose (2023-05-07 / May 2023 /
    2023). Dense cosine is blind to dates and proper nouns; BM25 over this matches
    them. Used only for the fused ranking, never returned to the answerer."""
    d = (e.occurred_at or "")[:10]
    forms = [d]
    if len(d) >= 7:
        try:
            forms += [f"{_MONTHS[int(d[5:7])]} {d[:4]}", d[:4]]
        except (ValueError, IndexError):
            pass
    return " ".join([_event_result(None, e)["text"], (e.raw_text or ""), *forms])


def _rank_events(embedder: Embedder, events: list[Event], question: str, k: int,
                 order: str = "time", append_raw: bool = False) -> list[Event]:
    """Top-k events for the question. order='time' returns them chronologically (for
    timeline display). order='relevance' is the RETRIEVAL path: it fuses the dense
    (embedding) ranking with a BM25 lexical ranking by UNION — take each ranker's
    top-k, reorder by best (min) component rank, keep k. The union includes the dense
    ranking's top-k, so lexical matches (dates and proper nouns the embedding misses)
    supplement retrieval without displacing dense hits. Read-side only."""
    from atomir.episodic.registry import cosine
    qv = embedder.embed_query(question)
    dense = sorted(range(len(events)), reverse=True,
                   key=lambda i: cosine(qv, embedder.embed_passage(
                       _event_result(None, events[i], append_raw)["text"])))
    if order != "relevance":
        return sorted([events[i] for i in dense[:k]], key=lambda e: e.order_key)
    scores = BM25([_lex_text(e) for e in events]).scores(question)
    bm25 = sorted(range(len(events)), key=lambda i: scores[i], reverse=True)
    dr = {i: r for r, i in enumerate(dense)}
    br = {i: r for r, i in enumerate(bm25)}
    fused = sorted(set(dense[:k]) | set(bm25[:k]), key=lambda i: min(dr[i], br[i]))
    return [events[i] for i in fused[:k]]


def _events_by_similarity(episodic: EpisodicStore, embedder: Embedder, user_id: str,
                          entity_id: str, question: str, k: int,
                          append_raw: bool = False) -> list[Event]:
    """Bounded fallback when no branch resolves: rank the entity's events by
    similarity to the question, top-k — far better than dumping the timeline."""
    return _rank_events(embedder, walk_chain(episodic, user_id, entity_id=entity_id),
                        question, k, order="relevance", append_raw=append_raw)


def episodic_search(facts: MemoryStore, episodic: EpisodicStore, llm: LLM,
                    embedder: Embedder, user_id: str, query: str, *, k: int = 6,
                    decompose: bool = True, hybrid: bool = True,
                    resolve_floor: float = 0.30, resolve_margin: float = 0.10,
                    planner_llm: LLM | None = None,
                    temporal_expand_checkpoints: str = "off") -> dict:
    """Typed decomposition + routing + dedup. Returns {subquestions,
    subquestion_types, results}."""
    if decompose:
        from atomir.ontology import branch_vocab
        plan = plan_typed(planner_llm or llm, query,
                          branch_vocab(episodic.branches(user_id)))
    else:
        plan = [{"text": query, "type": SEMANTIC, "entity_hint": None, "branch_hint": None}]

    # Surface lossy source clauses ONLY for a direct temporal lookup (one temporal
    # sub-question). Multi-hop/open-domain queries decompose into several parts
    # where the extra event text adds noise, so they keep the clean projection.
    pure_temporal = len(plan) == 1 and plan[0]["type"] == TEMPORAL

    results: list[dict] = []
    seen_facts: set[str] = set()   # fact ids already represented (dedup)
    branch_resolved = False        # a temporal sub-q hit a real chain walk
    fallback_used = False          # a temporal sub-q fell back to ranked events
    routes: list[dict] = []        # per sub-question diagnostics
    mechanisms: list[str] = []     # exact | relative-best | fallback | none

    for sq in plan:
        if sq["type"] == TEMPORAL:
            entity = episodic.entity_by_alias(user_id, sq["entity_hint"]) if sq["entity_hint"] else None
            if entity is None:  # temporal questions are about the user by default
                entity = episodic.entity_by_alias(user_id, "the user")
            entity_id = entity.entity_id if entity else None
            branch = (_resolve_branch(episodic, user_id, entity_id, sq["branch_hint"],
                                      embedder, resolve_floor, resolve_margin)
                      if entity_id else None)
            # If the planner hint didn't resolve (emergent names like "joined"
            # won't match "works_at"), try resolving from the sub-question text
            # itself, so the chain walk still fires without a pack.
            if branch is None and entity_id:
                branch = _resolve_branch(episodic, user_id, entity_id, sq["text"],
                                         embedder, resolve_floor, resolve_margin)
            if branch is not None:
                branch_resolved = True
                hint = (sq["branch_hint"] or "").strip().casefold()
                is_exact = bool(hint) and (branch.branch.casefold() == hint
                    or branch.branch.replace("_", " ").casefold() == hint
                    or hint in {a.casefold() for a in branch.aliases})
                mechanisms.append("exact" if is_exact else "relative-best")
                events = walk_chain(episodic, user_id, entity_id=entity_id, branch=branch.branch)
                # Rank the chain by relevance to the sub-question so the matching
                # event surfaces into a tight top-k cutoff (dates in the text keep
                # chronology); checkpoints stay pinned at the front.
                if temporal_expand_checkpoints == "on":
                    events = _expand_checkpoint_members(episodic, user_id, events)
                cps = [e for e in events if e.kind == "checkpoint"]
                rest = [e for e in events if e.kind != "checkpoint"]
                events = cps + _rank_events(embedder, rest, sq["text"], k,
                                            order="relevance", append_raw=pure_temporal)
            elif entity_id:
                # No branch resolved -> bounded semantic top-k over the entity's
                # timeline. When checkpoint expansion is on, the pool includes
                # absorbed members too so a chunk-summarised event can surface.
                fallback_used = True
                mechanisms.append("fallback")
                pool = walk_chain(episodic, user_id, entity_id=entity_id)
                if temporal_expand_checkpoints == "on":
                    pool = _expand_checkpoint_members(episodic, user_id, pool)
                events = _rank_events(embedder, pool, sq["text"], k,
                                      order="relevance", append_raw=pure_temporal)
            else:
                mechanisms.append("none")
                events = []
            for e in events:
                if e.fact_id:
                    seen_facts.add(e.fact_id)
                results.append(_event_result(episodic, e, append_raw=pure_temporal))
            routes.append({"type": TEMPORAL,
                           "route": "chain_walk" if branch is not None
                           else ("fallback" if entity_id else "none"),
                           "items": len(events)})
        else:  # current / semantic -> existing fact retrieval
            # In append-only mode superseded facts stay in the store with
            # is_current=False. Current-typed sub-Qs want the live state;
            # semantic sub-Qs benefit from seeing historical facts too.
            _only_current = sq["type"] != SEMANTIC
            found = atomic_search(facts, llm, embedder, user_id, sq["text"], k=k,
                                  decompose=False, hybrid=hybrid,
                                  only_current=_only_current)
            n_facts = 0
            for r in found["results"]:
                if r["id"] in seen_facts:  # already shown as a temporal event
                    continue
                seen_facts.add(r["id"])
                results.append({**r, "type": sq["type"]})
                n_facts += 1
            # H3 fix: semantic recall also searches the RAW episode text, which
            # keeps the user's original wording (projection state-phrasing can
            # drop keywords the query matches on, e.g. "hobby").
            ep_hits = 0
            if sq["type"] == SEMANTIC:
                for ep in _episode_matches(episodic, user_id, sq["text"], k):
                    results.append({"id": ep.id, "text": ep.text, "type": SEMANTIC,
                                    "source": "episode"})
                    ep_hits += 1
            routes.append({"type": sq["type"], "route": "facts",
                           "fact_hits": n_facts, "episode_hits": ep_hits})

    return {
        "subquestions": [sq["text"] for sq in plan],
        "subquestion_types": [sq["type"] for sq in plan],
        "results": results[: max(k, len(results))] if results else [],
        "branch_resolved": branch_resolved,
        "fallback_used": fallback_used,
        "resolve_mechanism": mechanisms[0] if mechanisms else None,
        "routes": routes,
    }


def _episode_matches(episodic: EpisodicStore, user_id: str, question: str, k: int):
    """Lexical (BM25) matches over raw episode texts — recovers the user's
    original wording that projection may have dropped."""
    episodes = episodic.episodes(user_id)
    if not episodes:
        return []
    bm25 = BM25([e.text for e in episodes])
    scored = [(s, episodes[i]) for i, s in enumerate(bm25.scores(question)) if s > 0]
    scored.sort(key=lambda t: t[0], reverse=True)
    return [ep for _, ep in scored[:k]]
