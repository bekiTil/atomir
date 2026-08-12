"""Project an event into the fact store.

Single extraction, dual projection: facts are DERIVED from events, never
extracted independently. The projection is deterministic:

- Phrasing: facts use STATE phrasing templated from the branch, never
  verb phrasing. `start`/`update` -> "The user {template} {value}"; `end` ->
  "The user no longer {template} {value}" (UPDATE to negated state, NEVER
  DELETE).
- Target: the branch's `current_fact_id` tells us
  exactly which fact to UPDATE, so a Beta->Acme transition can never leave both
  employers live — we don't rely on the fragile embedding similarity gate at all.
- Idempotency: an event already projected (flag set / fact_id linked) is
  skipped, so replay after a crash never double-writes.
- Only `happened` events project; `planned`/`hypothetical` are stored as events
  but produce no fact.
"""

from __future__ import annotations

from atomir.episodic.models import END, HAPPENED, BranchRecord, Event, value_phrase
from atomir.episodic.store_base import EpisodicStore
from atomir.providers.embedder_base import Embedder
from atomir.store_base import MemoryStore


# End/negation verbs that must NOT appear inside a STATE template — a state is
# present-tense ("works at"), never the act of ending it ("left works at").
_END_VERBS = {"left", "quit", "stopped", "resigned", "ended", "ceased", "dropped",
              "no", "longer", "resigned", "departed", "cancelled"}


def _clean_template(template: str) -> str:
    """Strip leading end-verbs and any object-type enum word a weak namer may have
    baked into a state template ('no longer left works at' -> 'no longer works at';
    'joined organization' -> 'joined')."""
    from atomir.ontology import OBJECT_TYPES

    words = [w for w in template.split() if w.casefold() not in OBJECT_TYPES]
    while words and words[0].casefold() in _END_VERBS:
        words.pop(0)
    return " ".join(words) or template


def state_text(branch: BranchRecord, event: Event) -> str:
    """Deterministic state phrasing for a projected fact. Negation
    renders ONLY from the (cleaned) branch state_template, never the event verb."""
    tmpl = _clean_template(branch.state_template)
    val = value_phrase(event.value, event.qualifier)
    if event.polarity == END:
        text = f"The user no longer {tmpl} {val}".strip()
    else:
        text = f"The user {tmpl} {val}".strip()
    # Render guard: the raw event verb must not leak a second time into the state.
    assert "no longer no longer" not in text, f"double negation in {text!r}"
    return text


def _values_match(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    a, b = a.casefold().strip(), b.casefold().strip()
    return a == b or a in b or b in a


def project_event(
    facts: MemoryStore,
    episodic: EpisodicStore,
    embedder: Embedder,
    user_id: str,
    event: Event,
    branch: BranchRecord,
    append_only: bool = False,
) -> dict:
    """Project one event; return {decision, fact, event_id, fact_id}."""
    # Idempotency: already projected -> do nothing, keep replay safe.
    if event.projected or event.fact_id:
        return {"decision": "NOOP", "reason": "already projected",
                "event_id": event.id, "fact": None}

    if event.modality != HAPPENED:
        event.projected = True
        episodic.update_event(event)
        return {"decision": "NOOP", "reason": f"modality={event.modality}",
                "event_id": event.id, "fact": None}

    current = (facts.get(user_id, branch.current_fact_id)
               if branch.current_fact_id else None)

    # An `end` only affects the current fact if it ends the CURRENT value; an end
    # of a value the user already moved on from is historical (logged as an
    # event, no fact change). This makes multi-event messages order-independent:
    # "left Beta" + "joined Acme" -> "works at Acme" regardless of event order.
    if event.polarity == END and not _values_match(branch.current_value, event.value):
        event.projected = True
        episodic.update_event(event)
        return {"decision": "NOOP", "reason": "historical end (not current value)",
                "event_id": event.id, "fact": None}

    text = state_text(branch, event)
    embedding = embedder.embed_passage(text)

    if append_only:
        # New fact record per write. Preserves the previous fact and its
        # embedding by marking it is_current=False in metadata, pushing its
        # id onto branch.fact_history_ids, and adding a new fact with
        # is_current=True. Multi/set cardinalities skip the demotion so all
        # values stay live at once.
        card = getattr(branch, "cardinality", "single")
        if current is not None and card == "single":
            old_meta = dict(current.get("metadata") or {})
            old_meta["is_current"] = False
            facts.update(user_id, current["id"], current["text"],
                          current.get("embedding") or embedding, metadata=old_meta)
            branch.fact_history_ids = list(getattr(branch, "fact_history_ids", []) or []) + [current["id"]]
        elif card in ("multi", "set") and current is not None:
            # Dedupe check for `set`: skip if a live fact with matching value
            # already exists on this branch.
            if card == "set":
                for fid in [branch.current_fact_id, *getattr(branch, "fact_history_ids", [])]:
                    if not fid:
                        continue
                    existing = facts.get(user_id, fid)
                    if existing and _values_match(
                            (existing.get("metadata") or {}).get("value"),
                            event.value):
                        # Duplicate value under set: no-op, event is still logged
                        event.projected = True
                        event.fact_id = existing["id"]
                        episodic.update_event(event)
                        return {"decision": "NOOP", "reason": "set duplicate",
                                "event_id": event.id, "fact": existing,
                                "fact_id": existing["id"]}

        fact = facts.add(user_id, text, embedding, metadata={
            "episodic": True, "entity_id": event.entity_id,
            "branch": event.branch, "value": event.value, "is_current": True})
        # For single cardinality, the newly-created fact IS the current one.
        # For multi/set, we still track the latest addition as `current_fact_id`
        # but the older ones stay live (not demoted).
        branch.current_fact_id = fact["id"]
        decision = "ADD"
    elif current is not None:
        fact = facts.update(user_id, branch.current_fact_id, text, embedding)
        decision = "UPDATE"
    else:
        fact = facts.add(user_id, text, embedding, metadata={
            "episodic": True, "entity_id": event.entity_id, "branch": event.branch})
        branch.current_fact_id = fact["id"]
        decision = "ADD"

    # Track the current value; an `end` clears it (the state has ceased).
    branch.current_value = None if event.polarity == END else event.value
    episodic.update_branch(branch)

    event.fact_id = fact["id"]
    event.projected = True
    episodic.update_event(event)
    return {"decision": decision, "fact": fact, "event_id": event.id,
            "fact_id": fact["id"]}
