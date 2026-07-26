"""Episodic write path: the orchestration MemoryService delegates to
when EPISODIC_ENABLED. One extraction call per add; facts are projected from the
resulting events (never extracted independently).

Write-ahead ordering: the event is appended (projected=False) BEFORE
its fact is written, and `replay()` re-projects anything left unprojected by a
crash — idempotently.
"""

from __future__ import annotations

from atomir.episodic.extractor import extract_events
from atomir.episodic.models import BranchRecord, Episode, Event, new_id, now_iso
from atomir.episodic.projection import project_event
from atomir.episodic.read import _event_result, episodic_search, walk_chain
from atomir.episodic.registry import BranchMatcher, EntityResolver
from atomir.episodic.store_base import EpisodicStore
from atomir.episodic.telemetry import CountingEmbedder, CountingLLM
from atomir.providers.embedder_base import Embedder
from atomir.providers.llm_base import LLM
from atomir.store_base import MemoryStore

_SUMMARY_SYSTEM = (
    "You summarize a segment of a personal-memory timeline branch into ONE "
    "concise line for a checkpoint. Respond ONLY with JSON: {\"summary\": \"...\"}"
)


class EpisodicMemory:
    def __init__(self, facts: MemoryStore, episodic: EpisodicStore, llm: LLM,
                 embedder: Embedder, *, branch_auto: float, branch_gray_low: float,
                 entity_v2: bool = False, entity_match_min: float = 0.60,
                 checkpoint_every: int = 0, ontology_pack: str = "",
                 resolve_floor: float = 0.30, resolve_margin: float = 0.10) -> None:
        self.facts = facts
        self.episodic = episodic
        self.checkpoint_every = checkpoint_every
        self.ontology_pack = ontology_pack
        self.resolve_floor = resolve_floor
        self.resolve_margin = resolve_margin
        # Counting proxies so add() can report real per-message cost.
        self.llm = CountingLLM(llm)
        self.embedder = CountingEmbedder(embedder)
        self.entities = EntityResolver(episodic, self.embedder, self.llm,
                                       v2=entity_v2, match_min=entity_match_min)
        self.matcher = BranchMatcher(episodic, self.llm, self.embedder,
                                     auto=branch_auto, gray_low=branch_gray_low)

    def add(self, user_id: str, text: str) -> dict:
        recorded_at = now_iso()
        self.llm.reset()
        self.embedder.reset()
        # Optional pack: pre-seed the registry for a known domain on first write.
        if self.ontology_pack:
            from atomir.ontology import apply_pack, get_pack
            seeds = get_pack(self.ontology_pack)
            if seeds:
                self_entity, _ = self.entities.resolve(user_id, "the user", "person")
                if not self.episodic.branches(user_id, self_entity.entity_id):
                    apply_pack(self.episodic, user_id, self_entity.entity_id, seeds)
        # Registry feedback (general-purpose): give extraction the user's existing
        # branches so it reuses them instead of inventing variants.
        from atomir.ontology import format_registry
        registry = format_registry(self.episodic.branches(user_id))
        episode = Episode(id=new_id("ep"), user_id=user_id, text=text, created_at=recorded_at)
        self.episodic.add_episode(episode)

        operations: list[dict] = []
        facts_out: list[dict] = []
        events_out: list[dict] = []
        touched: set[tuple[str, str]] = set()
        for rev in extract_events(self.llm, text, recorded_at, registry=registry):
            subject, value = rev["subject"], rev["value"]
            subj_entity, _ = self.entities.resolve(user_id, subject, entity_type=rev["subject_type"])
            value_entity_id = None
            if value:
                val_entity, _ = self.entities.resolve(user_id, value, entity_type=rev["object_type"])
                value_entity_id = val_entity.entity_id

            match = self.matcher.match(user_id, subj_entity.entity_id, rev["verb_phrase"],
                                       rev["subject_type"], rev["object_type"], {subject, value},
                                       value=value, polarity=rev["polarity"])
            branch = match["branch"]

            event = Event(
                id=new_id("ev"), user_id=user_id, entity_id=subj_entity.entity_id,
                branch=branch.branch, value=value, raw_text=rev["raw_text"],
                polarity=rev["polarity"], recorded_at=recorded_at,
                value_entity_id=value_entity_id, occurred_at=rev["occurred_at"],
                modality=rev["modality"], episode_id=episode.id, projected=False,
            )
            self.episodic.append_event(event)  # write-ahead: event before fact
            if rev.get("corrects_hint"):
                self._apply_correction(user_id, event, rev["corrects_hint"])
            op = project_event(self.facts, self.episodic, self.embedder, user_id, event, branch)
            operations.append(op)
            events_out.append(event.to_dict())
            if op.get("fact"):
                facts_out.append(op["fact"])
            touched.add((event.entity_id, branch.branch))

        for entity_id, branch_name in touched:
            self._maybe_checkpoint(user_id, entity_id, branch_name)

        return {"episode_id": episode.id, "events": events_out,
                "operations": operations, "facts": facts_out,
                "diagnostics": {  # real per-add cost, not "~1 call"
                    "extraction_calls": self.llm.c.extraction,
                    "judge_calls": self.llm.c.judge,
                    "embedding_calls": self.embedder.c.embedding,
                    "total_llm_tokens": self.llm.c.tokens,  # approximate (chars/4)
                }}

    def search(self, user_id: str, query: str, k: int = 6, decompose: bool = True,
               hybrid: bool = True) -> dict:
        """Typed decomposition + routing + dedup."""
        return episodic_search(self.facts, self.episodic, self.llm, self.embedder,
                               user_id, query, k=k, decompose=decompose, hybrid=hybrid,
                               resolve_floor=self.resolve_floor,
                               resolve_margin=self.resolve_margin)

    def timeline(self, user_id: str, entity: str | None = None, branch: str | None = None,
                 since: str | None = None, until: str | None = None) -> list[dict]:
        """Ordered events, optionally filtered by entity (name/alias) + branch."""
        entity_id = None
        if entity:
            rec = self.episodic.entity_by_alias(user_id, entity)
            entity_id = rec.entity_id if rec else "__nomatch__"
        branch_canonical = None
        if branch:
            hint = branch.strip().casefold()
            for b in self.episodic.branches(user_id, entity_id):
                if b.branch.casefold() == hint or b.branch.replace("_", " ").casefold() == hint \
                        or hint in {a.casefold() for a in b.aliases}:
                    branch_canonical = b.branch
                    break
            if branch_canonical is None:
                branch_canonical = branch  # let the store filter (may be empty)
        events = walk_chain(self.episodic, user_id, entity_id=entity_id,
                            branch=branch_canonical, since=since, until=until)
        return [_event_result(self.episodic, e) for e in events]

    def merge_entities(self, user_id: str, keep_id: str, absorb_id: str) -> bool:
        """Explicit entity merge: re-parent all of `absorb_id`'s
        events and branches onto `keep_id`, union aliases, delete the absorbed
        record. Never called automatically — over-merge is forbidden, so merges
        are a deliberate maintenance op. Events re-parented to keep_id interleave
        in occurred_at order automatically on read.
        """
        keep = self.episodic.get_entity(user_id, keep_id)
        absorb = self.episodic.get_entity(user_id, absorb_id)
        if keep is None or absorb is None or keep_id == absorb_id:
            return False

        # Union aliases into the surviving entity.
        for a in absorb.aliases:
            if a.casefold() not in {x.casefold() for x in keep.aliases}:
                keep.aliases.append(a)
        self.episodic.update_entity(keep)

        # Re-parent every event that referenced the absorbed entity (as subject
        # or as object/value).
        for e in self.episodic.events(user_id):
            changed = False
            if e.entity_id == absorb_id:
                e.entity_id, changed = keep_id, True
            if e.value_entity_id == absorb_id:
                e.value_entity_id, changed = keep_id, True
            if changed:
                self.episodic.update_event(e)

        # Re-parent branches; merge into a same-named keep branch when present.
        keep_branches = {b.branch: b for b in self.episodic.branches(user_id, keep_id)}
        for b in self.episodic.branches(user_id, absorb_id):
            if b.branch in keep_branches:
                kb = keep_branches[b.branch]
                for a in b.aliases:
                    if a.casefold() not in {x.casefold() for x in kb.aliases}:
                        kb.aliases.append(a)
                self.episodic.update_branch(kb)
                self.episodic.delete_branch(user_id, absorb_id, b.branch)
            else:
                self.episodic.delete_branch(user_id, absorb_id, b.branch)
                moved = BranchRecord(
                    branch=b.branch, user_id=user_id, entity_id=keep_id,
                    description=b.description, state_template=b.state_template,
                    aliases=list(b.aliases), current_fact_id=b.current_fact_id,
                    current_value=b.current_value)
                self.episodic.add_branch(moved)
                keep_branches[b.branch] = moved

        self.episodic.delete_entity(user_id, absorb_id)
        return True

    def reset(self, user_id: str) -> bool:
        """Clear facts AND all episodic data for the user."""
        cleared_facts = self.facts.clear(user_id)
        cleared_ep = self.episodic.clear(user_id)
        return cleared_facts or cleared_ep

    # --- cascade deletes ----------------------------
    def _redact_orphan_episodes(self, user_id: str, episode_ids) -> None:
        """Delete episodes that no surviving event references anymore."""
        live = {e.episode_id for e in self.episodic.events(user_id)}
        for ep_id in {e for e in episode_ids if e}:
            if ep_id not in live:
                self.episodic.delete_episode(user_id, ep_id)

    def _rederive_branch(self, user_id: str, entity_id: str, branch: str) -> None:
        """Rebuild a branch's projected fact from its SURVIVING events (in order),
        after some events were removed. Deletes the stale fact and replays."""
        rec = self.episodic.get_branch(user_id, entity_id, branch)
        if rec is None:
            return
        if rec.current_fact_id:
            self.facts.delete(user_id, rec.current_fact_id)
        rec.current_fact_id = None
        rec.current_value = None
        self.episodic.update_branch(rec)
        events = [e for e in self.episodic.events(user_id, entity_id=entity_id, branch=branch)
                  if e.kind == "event" and not e.checkpointed]
        for e in events:
            e.projected = False
            e.fact_id = None
            self.episodic.update_event(e)
        for e in events:  # replay in occurred_at order rebuilds current_fact_id
            project_event(self.facts, self.episodic, self.embedder, user_id, e, rec)

    def delete_fact(self, user_id: str, fact_id: str) -> bool:
        """Delete a fact AND its source events, redacting orphaned episodes and
        clearing any branch that pointed at it (cascade, one direction)."""
        source = [e for e in self.episodic.events(user_id) if e.fact_id == fact_id]
        removed = self.facts.delete(user_id, fact_id)
        episodes = set()
        for e in source:
            episodes.add(e.episode_id)
            self.episodic.delete_event(user_id, e.id)
        for b in self.episodic.branches(user_id):
            if b.current_fact_id == fact_id:
                b.current_fact_id = None
                b.current_value = None
                self.episodic.update_branch(b)
        self._redact_orphan_episodes(user_id, episodes)
        return removed or bool(source)

    def delete_event(self, user_id: str, event_id: str) -> bool:
        """Delete one event; re-derive its branch's fact from what remains
        (cascade, the other direction)."""
        e = self.episodic.get_event(user_id, event_id)
        if e is None:
            return False
        self.episodic.delete_event(user_id, event_id)
        self._rederive_branch(user_id, e.entity_id, e.branch)
        self._redact_orphan_episodes(user_id, {e.episode_id})
        return True

    def forget_entity(self, user_id: str, entity_id: str) -> bool:
        """Forget everything about an entity: delete every event referencing it
        (as subject or value), its own branches + facts, then the entity — and
        re-derive any OTHER branch whose facts depended on those events. Episodes
        that only referenced it are deleted; episodes that ALSO hold other events
        have the entity's name REDACTED from their raw text (so forget is real
        even in the raw layer)."""
        entity = self.episodic.get_entity(user_id, entity_id)
        if entity is None:
            return False
        names = {n for n in [*entity.aliases, entity.canonical_name] if n and n.strip()}
        events = [e for e in self.episodic.events(user_id)
                  if e.entity_id == entity_id or e.value_entity_id == entity_id]
        episodes = {e.episode_id for e in events}
        # branches of OTHER entities that lose events -> re-derive after deletion.
        foreign = {(e.entity_id, e.branch) for e in events if e.entity_id != entity_id}
        for e in events:
            self.episodic.delete_event(user_id, e.id)
        for b in self.episodic.branches(user_id, entity_id):
            if b.current_fact_id:
                self.facts.delete(user_id, b.current_fact_id)
            self.episodic.delete_branch(user_id, entity_id, b.branch)
        for eid, br in foreign:
            self._rederive_branch(user_id, eid, br)
        self.episodic.delete_entity(user_id, entity_id)
        self._redact_episodes(user_id, episodes, names)
        return True

    def _redact_episodes(self, user_id: str, episode_ids, names) -> None:
        """Delete episodes orphaned by the forget; then scrub the forgotten
        entity's name(s) from EVERY remaining episode's raw text — the name can
        appear in a message that produced no event about it (e.g. the 'forget X'
        instruction itself), so redaction must scan all episodes, not just the
        ones that held the entity's events."""
        import re

        touched = {e for e in episode_ids if e}
        live = {e.episode_id for e in self.episodic.events(user_id)}
        for ep_id in touched:
            if ep_id not in live:
                self.episodic.delete_episode(user_id, ep_id)
        if not names:
            return
        pattern = re.compile(r"\b(" + "|".join(re.escape(n) for n in
                             sorted(names, key=len, reverse=True)) + r")\b", re.IGNORECASE)
        for ep in self.episodic.episodes(user_id):
            redacted = pattern.sub("[forgotten]", ep.text)
            if redacted != ep.text:
                self.episodic.delete_episode(user_id, ep.id)
                ep.text = redacted
                self.episodic.add_episode(ep)

    def _apply_correction(self, user_id: str, event: Event, hint: str) -> None:
        """Link `event` to the earlier event it corrects and mark that one
        superseded (a correction, not a real state change; R12). No double-count:
        projection is a branch-keyed upsert, so the correction just updates the
        current fact. The superseded event drops out of the chain walk."""
        hint_words = {w for w in hint.casefold().split() if len(w) > 2}
        prior = [e for e in self.episodic.events(user_id, entity_id=event.entity_id,
                                                 branch=event.branch)
                 if e.id != event.id and not e.superseded]
        for cand in reversed(prior):  # most recent match wins
            words = {w for w in cand.value.casefold().split() if len(w) > 2}
            if hint_words & words or hint.casefold() in cand.raw_text.casefold():
                cand.superseded = True
                self.episodic.update_event(cand)
                event.corrects = cand.id
                self.episodic.update_event(event)
                return

    def _maybe_checkpoint(self, user_id: str, entity_id: str, branch: str) -> None:
        """When a branch's live event count exceeds checkpoint_every, summarize
        its oldest segment into one checkpoint node. Raw events are
        marked checkpointed (kept, not deleted); the chain walk then reads the
        checkpoint plus the recent tail."""
        if self.checkpoint_every <= 0:
            return
        live = [e for e in self.episodic.events(user_id, entity_id=entity_id, branch=branch)
                if e.kind == "event" and not e.checkpointed and not e.superseded]
        if len(live) <= self.checkpoint_every:
            return
        keep_tail = max(1, self.checkpoint_every // 2)
        segment = live[:-keep_tail]
        if not segment:
            return
        lines = "; ".join(f"{e.polarity} {e.value}" for e in segment)
        res = self.llm.chat_json(_SUMMARY_SYSTEM, f"BRANCH {branch}:\n{lines}")
        summary = str((res or {}).get("summary") or lines)[:500]
        self.episodic.append_event(Event(
            id=new_id("ev"), user_id=user_id, entity_id=entity_id, branch=branch,
            value=summary, raw_text=summary, polarity="update", recorded_at=now_iso(),
            occurred_at=segment[-1].occurred_at or segment[-1].recorded_at,
            kind="checkpoint", projected=True))
        for e in segment:
            e.checkpointed = True
            self.episodic.update_event(e)

    def repair_branches(self, user_id: str) -> int:
        """Healing pass (B1): re-home end events that sit in a branch with no
        matching start-value onto the branch that DID start that value, then drop
        now-empty stray branches and re-derive affected facts. Fixes stores split
        by inconsistent naming before value-anchoring existed. Returns #moved."""
        from atomir.episodic.registry import _type_compat, _value_match

        def starts(b):
            return [e for e in self.episodic.events(user_id, entity_id=b.entity_id,
                                                    branch=b.branch) if e.polarity == "start"]

        branches = self.episodic.branches(user_id)
        moved = 0
        for b in branches:
            b_starts = starts(b)
            for e in list(self.episodic.events(user_id, entity_id=b.entity_id, branch=b.branch)):
                if e.polarity != "end" or any(_value_match(e.value, s.value) for s in b_starts):
                    continue
                targets = [t for t in branches
                           if t.branch != b.branch and t.entity_id == b.entity_id
                           and _type_compat(t.object_type, b.object_type)
                           and any(_value_match(e.value, s.value) for s in starts(t))]
                if len(targets) == 1:
                    e.branch = targets[0].branch
                    self.episodic.update_event(e)
                    moved += 1

        # Drop empty stray branches (+ their facts); re-derive the rest.
        for b in self.episodic.branches(user_id):
            if not self.episodic.events(user_id, entity_id=b.entity_id, branch=b.branch):
                if b.current_fact_id:
                    self.facts.delete(user_id, b.current_fact_id)
                self.episodic.delete_branch(user_id, b.entity_id, b.branch)
            else:
                self._rederive_branch(user_id, b.entity_id, b.branch)
        return moved

    def backfill(self, user_id: str) -> int:
        """Non-destructive migration: for each existing fact with no
        linked event, synthesize ONE `update` event (occurred_at=None,
        recorded_at=fact.created_at) so timelines start populated. Idempotent:
        facts already backed by an event are skipped. Facts are left untouched.
        Returns the number of events created."""
        linked = {e.fact_id for e in self.episodic.events(user_id) if e.fact_id}
        self_entity, _ = self.entities.resolve(user_id, "the user", entity_type="person")
        created = 0
        for fact in self.facts.all(user_id):
            if fact["id"] in linked:
                continue
            when = fact.get("created_at") or now_iso()
            episode = Episode(id=new_id("ep"), user_id=user_id, text=fact["text"],
                              created_at=when)
            self.episodic.add_episode(episode)
            self.episodic.append_event(Event(
                id=new_id("ev"), user_id=user_id, entity_id=self_entity.entity_id,
                branch="legacy", value=fact["text"], raw_text=fact["text"],
                polarity="update", recorded_at=when, occurred_at=None,
                episode_id=episode.id, fact_id=fact["id"], projected=True))
            created += 1
        return created

    def replay(self, user_id: str) -> int:
        """Re-project events left unprojected by a crash. Idempotent:
        already-projected events are skipped. Returns how many were projected."""
        projected = 0
        for event in self.episodic.unprojected_events(user_id):
            branch = self.episodic.get_branch(user_id, event.entity_id, event.branch)
            if branch is None:  # event logged before its branch persisted; skip
                continue
            op = project_event(self.facts, self.episodic, self.embedder, user_id, event, branch)
            if op["decision"] != "NOOP" or op.get("reason") != "already projected":
                projected += 1
        return projected
