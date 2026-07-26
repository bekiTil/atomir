"""Entity and branch resolution.

Entity resolution v1 is deliberately conservative: exact casefold alias match,
else a NEW entity. Never auto-merge (over-merge is forbidden; under-merge is the
accepted failure direction — see `merge_entities`).

Branch matching is the hard part: same-branch and different-branch embedding
similarities OVERLAP, and shared proper nouns inflate cross-branch similarity —
so:
  - names are STRIPPED before embedding: embed "reports to (person ->
    person)", never "reports to Dana";
  - assignment is THREE-ZONE: >= AUTO auto-assign; [GRAY_LOW, AUTO) ask
    the LLM judge; < GRAY_LOW propose a new branch but still let the LLM confirm.
Embeddings NOMINATE; the LLM DECIDES.
"""

from __future__ import annotations

import math
import re

from atomir.episodic.models import BranchRecord, EntityRecord, new_id
from atomir.episodic.store_base import EpisodicStore
from atomir.providers.embedder_base import Embedder
from atomir.providers.llm_base import LLM

_JUDGE_SYSTEM = (
    "You are a branch matcher for a personal-memory system. A branch is ONE "
    "specific relationship type for an entity, e.g. works_at, lives_in, "
    "reports_to. Given a NEW predicate phrase and the entity's EXISTING branches, "
    "decide which existing branch it belongs to, or that it is a NEW branch.\n"
    "Default STRONGLY to NEW. Only pick an existing branch when the new phrase is "
    "the SAME relationship — a paraphrase or the opposite DIRECTION of it (joined "
    "vs left the same job; started vs stopped the same activity). Different KINDS "
    "of relationship are ALWAYS different branches: employment, residence, hobby, "
    "ownership, family, friendship are each their own branch — never merge them.\n"
    'Respond ONLY with JSON: {"branch": "<existing canonical predicate, or NEW>"}'
)

_NAME_SYSTEM = (
    "You name a NEW relationship branch for a knowledge-graph memory of a person. "
    "Given a predicate phrase and its subject/object types, return the CANONICAL "
    "relationship predicate as a standard snake_case knowledge-graph relation — "
    "name the RELATIONSHIP TYPE, never the surface verb. 'joined' (-> "
    "organization) is employment, so branch=works_at (NOT joined); 'moved to' "
    "(-> place) is lives_in. Also return a present-tense state_template used as "
    "'The user {template} {value}', a one-line relationship description, and 3-6 "
    "aliases (surface verbs for this relation) that MUST include the given phrase.\n"
    "Examples:\n"
    'phrase "joined" (person -> organization) -> {"branch": "works_at", '
    '"state_template": "works at", "description": "employment relationship between a '
    'person and an organization", "aliases": ["joined", "left", "quit", "hired at", '
    '"works for"]}\n'
    'phrase "moved to" (person -> place) -> {"branch": "lives_in", '
    '"state_template": "lives in", "description": "the place a person lives", '
    '"aliases": ["moved to", "relocated to", "lives in", "resides in"]}\n'
    'Respond ONLY with JSON: {"branch": "...", "state_template": "...", '
    '"description": "...", "aliases": ["..."]}'
)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def strip_names(phrase: str, names) -> str:
    """Remove resolved entity names/mentions from a verb phrase, so
    shared proper nouns don't drive branch similarity. Case-insensitive, word
    boundaries; whitespace collapsed."""
    out = phrase
    for name in sorted({n for n in names if n and n.strip()}, key=len, reverse=True):
        out = re.sub(rf"\b{re.escape(name)}\b", " ", out, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", out).strip()


_ENTITY_JUDGE_SYSTEM = (
    "You are an entity resolver for a personal-memory system. Decide whether a "
    "new MENTION refers to the SAME real-world entity as an EXISTING one (e.g. 'D' "
    "and 'Dana', 'my manager' and 'Dana'). Only say yes if they clearly co-refer; "
    "distinct people who share a name are NOT the same. Respond ONLY with JSON: "
    '{"same": true|false}'
)


class EntityResolver:
    """Resolve a mention to an entity.

    v1 (default): exact casefold alias match, else a NEW entity — never merges,
    under-merges by design. v2 (ENTITY_V2): if no alias matches, embed the
    mention, shortlist existing entities by similarity, and let the LLM confirm
    an ambiguous co-reference before attaching the mention as an alias. Still
    never merges two EXISTING entities (that is `merge_entities`).
    """

    def __init__(self, store: EpisodicStore, embedder: Embedder | None = None,
                 llm: LLM | None = None, *, v2: bool = False, match_min: float = 0.60) -> None:
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.v2 = v2
        self.match_min = match_min

    def resolve(self, user_id: str, mention: str, entity_type: str | None = None):
        """Return (EntityRecord, created: bool)."""
        mention = mention.strip()
        existing = self.store.entity_by_alias(user_id, mention)
        if existing is not None:
            return existing, False

        if self.v2 and self.embedder is not None and self.llm is not None:
            match = self._resolve_v2(user_id, mention, entity_type)
            if match is not None:
                return match, False

        entity = EntityRecord(
            entity_id=new_id("ent"), user_id=user_id,
            canonical_name=mention, aliases=[mention], entity_type=entity_type,
        )
        self.store.add_entity(entity)
        return entity, True

    def _resolve_v2(self, user_id: str, mention: str, entity_type: str | None):
        candidates = self.store.entities(user_id)
        if not candidates:
            return None
        mvec = self.embedder.embed_passage(mention)
        scored = [(max(cosine(mvec, self.embedder.embed_passage(a)) for a in c.aliases or [c.canonical_name]), c)
                  for c in candidates]
        top_score, top = max(scored, key=lambda t: t[0])
        if top_score < self.match_min:
            return None
        aliases = ", ".join(top.aliases)
        res = self.llm.chat_json(
            _ENTITY_JUDGE_SYSTEM,
            f"MENTION: {mention} ({entity_type or 'unknown type'})\n"
            f"EXISTING ENTITY: {top.canonical_name} (aliases: {aliases})")
        if not (isinstance(res, dict) and res.get("same")):
            return None
        if mention.casefold() not in {a.casefold() for a in top.aliases}:
            top.aliases.append(mention)
            self.store.update_entity(top)
        return top


class BranchMatcher:
    """Three-zone, name-stripped branch resolver. Returns the branch to use and
    telemetry about how it was decided."""

    def __init__(self, store: EpisodicStore, llm: LLM, embedder: Embedder,
                 *, auto: float, gray_low: float) -> None:
        self.store = store
        self.llm = llm
        self.embedder = embedder
        self.auto = auto
        self.gray_low = gray_low

    def _embed_text(self, verb_phrase: str, subject_type: str, object_type: str,
                    names) -> str:
        return f"{strip_names(verb_phrase, names)} ({subject_type} -> {object_type})"

    def _branch_embed_texts(self, branch: BranchRecord, subject_type: str,
                            object_type: str) -> list[str]:
        ctx = f"({subject_type} -> {object_type})"
        alias_phrases = [f"{a} {ctx}" for a in branch.aliases]
        return [f"{branch.branch.replace('_', ' ')} {ctx}", branch.description, *alias_phrases]

    def match(self, user_id: str, entity_id: str, verb_phrase: str,
              subject_type: str, object_type: str, names, *,
              value: str = "", polarity: str = "") -> dict:
        """Resolve `verb_phrase` to a branch for this entity.

        Returns {"branch": BranchRecord, "created": bool, "zone": str,
                 "judge_used": bool, "embedding_calls": int}.
        """
        existing = self.store.branches(user_id, entity_id)
        stripped = strip_names(verb_phrase, names).casefold()

        # 1. exact alias / canonical match — no embedding, no LLM.
        for b in existing:
            if stripped == b.branch.replace("_", " ") or stripped in {
                strip_names(a, names).casefold() for a in b.aliases
            }:
                return {"branch": b, "created": False, "zone": "exact",
                        "judge_used": False, "embedding_calls": 0}

        # Merge guard: a branch can only absorb a verb whose OBJECT TYPE matches
        # (a person->city "lives in" must never merge into a person->organization
        # "works at"). Only same-typed branches are merge-eligible; the rest can't
        # be confused no matter what a weak judge says. Type compare is lenient so
        # weak-model variants ("org" vs "organization") don't spuriously split.
        typed = [b for b in existing if _type_compat(b.object_type, object_type)]

        # value-anchored resolution for end/update. The thing being ended must
        # already exist as a started state; its VALUE identifies the branch, so we
        # never rely on matching "left" to "joined" by verb (which weak models
        # fail). Deterministic and model-independent.
        if polarity in ("end", "update") and value:
            anchored = [b for b in typed if any(
                e.polarity == "start" and _value_match(value, e.value)
                for e in self.store.events(user_id, entity_id=entity_id, branch=b.branch))]
            if len(anchored) == 1:
                self._record_provisional_alias(anchored[0], verb_phrase, names)
                return {"branch": anchored[0], "created": False, "zone": "value_anchored",
                        "judge_used": False, "embedding_calls": 0}
            if len(anchored) > 1:
                chosen = self._judge(verb_phrase, anchored)
                for b in anchored:
                    if b.branch == chosen:
                        self._record_provisional_alias(b, verb_phrase, names)
                        return {"branch": b, "created": False, "zone": "value_anchored",
                                "judge_used": True, "embedding_calls": 0}
            # zero matches -> fall through to normal verb matching (a user can
            # report an ending the system never saw start).

        # 2. embeddings NOMINATE (names stripped).
        emb_calls = 0
        top, top_score = None, 0.0
        if typed:
            cand_vec = self.embedder.embed_passage(
                self._embed_text(verb_phrase, subject_type, object_type, names))
            emb_calls += 1
            for b in typed:
                for t in self._branch_embed_texts(b, subject_type, object_type):
                    score = cosine(cand_vec, self.embedder.embed_passage(t))
                    emb_calls += 1
                    if score > top_score:
                        top, top_score = b, score

        # 3. three-zone decision.
        if top is not None and top_score >= self.auto:
            self._record_alias(top, verb_phrase, names)
            return {"branch": top, "created": False, "zone": "auto",
                    "judge_used": False, "embedding_calls": emb_calls}

        judge_used = False
        chosen_canonical = None
        if typed:  # gray zone OR below -> let the LLM decide among SAME-TYPE branches
            judge_used = True
            chosen_canonical = self._judge(verb_phrase, typed)

        if chosen_canonical and chosen_canonical.upper() != "NEW":
            for b in typed:
                if b.branch == chosen_canonical:
                    self._record_alias(b, verb_phrase, names)
                    return {"branch": b, "created": False, "zone": "judge",
                            "judge_used": judge_used, "embedding_calls": emb_calls}

        # 4. NEW branch — LLM names the canonical predicate + state template.
        new_branch = self._name_new_branch(user_id, entity_id, verb_phrase,
                                            subject_type, object_type, names)
        return {"branch": new_branch, "created": True, "zone": "new",
                "judge_used": judge_used, "embedding_calls": emb_calls}

    # --- helpers ----------------------------------------------------------
    def _record_alias(self, branch: BranchRecord, verb_phrase: str, names) -> None:
        alias = strip_names(verb_phrase, names)
        if alias and alias.casefold() not in {a.casefold() for a in branch.aliases}:
            branch.aliases.append(alias)
            self.store.update_branch(branch)

    def _record_provisional_alias(self, branch: BranchRecord, verb_phrase: str, names) -> None:
        """Quarantine: keep a value-anchored verb OUT of the confirmed alias
        set so it can't drive future exact matches on its own."""
        alias = strip_names(verb_phrase, names)
        known = {a.casefold() for a in branch.aliases + branch.provisional_aliases}
        if alias and alias.casefold() not in known:
            branch.provisional_aliases.append(alias)
            self.store.update_branch(branch)

    def _judge(self, verb_phrase: str, existing: list[BranchRecord]) -> str:
        listing = "\n".join(f"- {b.branch}: {b.description}" for b in existing)
        user = f"NEW PREDICATE PHRASE:\n{verb_phrase}\n\nEXISTING BRANCHES:\n{listing}"
        res = self.llm.chat_json(_JUDGE_SYSTEM, user)
        return str(res.get("branch", "NEW"))

    def _name_new_branch(self, user_id: str, entity_id: str, verb_phrase: str,
                         subject_type: str, object_type: str, names) -> BranchRecord:
        user = (f"PREDICATE PHRASE: {verb_phrase}\n"
                f"SUBJECT TYPE: {subject_type}\nOBJECT TYPE: {object_type}")
        # Reuse-first feedback: show the entity's existing branches so the model
        # returns an established canonical when the phrase fits one.
        from atomir.ontology import format_registry
        existing_branches = self.store.branches(user_id, entity_id)
        if existing_branches:
            user += ("\n\nEXISTING branches — return one of these canonical names if "
                     "the phrase fits it, else a NEW snake_case name:\n"
                     + format_registry(existing_branches))
        res = self.llm.chat_json(_NAME_SYSTEM, user)
        canonical = str(res.get("branch") or "").strip() or _fallback_predicate(verb_phrase, names)
        # Sanitize: strip any object-type enum word that leaked in ("joined
        # organization" -> "joined"), then reject placeholders/empties.
        state_template = _strip_type_words(str(res.get("state_template") or "").strip())
        if not state_template or "{" in state_template:
            state_template = (_strip_type_words(strip_names(verb_phrase, names))
                              or canonical.replace("_", " "))
        description = str(res.get("description") or "").strip() or f"{canonical} ({subject_type} -> {object_type})"
        # Aliases: the surface verb plus the namer's aliases (dedup, name-stripped).
        alias_src = [verb_phrase, *(a for a in (res.get("aliases") or []) if isinstance(a, str))]
        aliases: list[str] = []
        for a in alias_src:
            a = strip_names(a, names)
            if a and a.casefold() not in {x.casefold() for x in aliases}:
                aliases.append(a)
        branch = BranchRecord(
            branch=canonical, user_id=user_id, entity_id=entity_id,
            description=description, state_template=state_template,
            object_type=object_type, aliases=aliases or [strip_names(verb_phrase, names)],
        )
        # Guard: if the LLM named a predicate that already exists, reuse it.
        existing = self.store.get_branch(user_id, entity_id, canonical)
        if existing is not None:
            self._record_alias(existing, verb_phrase, names)
            return existing
        self.store.add_branch(branch)
        return branch


def _strip_type_words(template: str) -> str:
    """Remove object-type enum words a weak namer baked into a state template
    ('joined organization' -> 'joined'), so state phrasing stays clean."""
    from atomir.ontology import OBJECT_TYPES
    words = [w for w in template.split() if w.casefold() not in OBJECT_TYPES]
    return " ".join(words)


def _value_match(a: str, b: str) -> bool:
    """Lenient value equality for anchoring end events to their start branch:
    casefold, strip punctuation, accept containment or difflib ratio >= 0.75, so
    'Beta' matches 'Beta Inc'."""
    import difflib
    a = re.sub(r"[^\w\s]", " ", (a or "").casefold())
    a = re.sub(r"\s+", " ", a).strip()
    b = re.sub(r"[^\w\s]", " ", (b or "").casefold())
    b = re.sub(r"\s+", " ", b).strip()
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.75


def _type_compat(a: str, b: str) -> bool:
    """Lenient object-type compatibility: empty matches anything; otherwise equal
    or one a prefix of the other (so 'org' and 'organization' don't split)."""
    a, b = (a or "").lower().strip(), (b or "").lower().strip()
    return not a or not b or a == b or a.startswith(b) or b.startswith(a)


def _fallback_predicate(verb_phrase: str, names) -> str:
    """Deterministic snake_case predicate if the LLM returns nothing usable."""
    words = re.sub(r"[^\w\s]", " ", strip_names(verb_phrase, names).lower()).split()
    return "_".join(words[:3]) or "related_to"
