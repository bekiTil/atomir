"""Object/qualifier split (extraction steps 1-3): the object stays in `value`
and a time-quantity goes into the new `qualifier` field, so 'friends for 4 years'
no longer sacrifices 'friends'. Read/answer output is unchanged for now — the
field is captured but not yet surfaced (kept behind current behavior)."""

from __future__ import annotations

from atomir.embeddings.fake import FakeEmbedder
from atomir.episodic.engine import EpisodicMemory
from atomir.episodic.extractor import extract_events
from atomir.episodic.json_store import JsonEpisodicStore
from atomir.episodic.models import Event, new_id, now_iso
from atomir.stores.json_store import JsonMemoryStore
from doubles import ScriptedLLM


class _OneShotLLM(ScriptedLLM):
    """extract returns a fixed payload; branch_name gives a stable branch."""


def _mem(tmp_path, extract, branch_names):
    facts = JsonMemoryStore(path=str(tmp_path / "f.json"))
    ep = JsonEpisodicStore(path=str(tmp_path / "e.json"))
    llm = ScriptedLLM(responses={"extract": extract, "branch_name": branch_names})
    mem = EpisodicMemory(facts, ep, llm, FakeEmbedder(),
                         branch_auto=0.8, branch_gray_low=0.4)
    return mem, facts, ep


# ---- the Event schema carries the new field, defaulted (old records load) ----

def test_event_defaults_qualifier_none():
    e = Event(id=new_id("ev"), user_id="u", entity_id="ent", branch="b",
              value="friends", raw_text="x", polarity="start", recorded_at=now_iso())
    assert e.qualifier is None
    # round-trips through the store dict form
    assert Event.from_dict(e.to_dict()).qualifier is None


def test_from_dict_reads_qualifier():
    d = {"id": "ev1", "user_id": "u", "entity_id": "ent", "branch": "b",
         "value": "friends", "raw_text": "x", "polarity": "start",
         "recorded_at": now_iso(), "qualifier": "4 years"}
    assert Event.from_dict(d).qualifier == "4 years"


# ---- extraction normalization: object in value, span in qualifier ----

def test_extractor_keeps_object_and_captures_qualifier():
    llm = ScriptedLLM(responses={"extract": [{"events": [{
        "verb_phrase": "is related to", "value": "friends", "qualifier": "4 years",
        "subject": "the user", "subject_type": "person", "object_type": "person",
        "polarity": "start", "modality": "happened", "occurred_at": None,
        "raw_text": "I've known these friends for 4 years"}]}]})
    evs = extract_events(llm, "I've known these friends for 4 years", now_iso())
    assert len(evs) == 1
    assert evs[0]["value"] == "friends"        # object preserved
    assert evs[0]["qualifier"] == "4 years"    # span captured separately


def test_extractor_qualifier_defaults_none_when_absent():
    llm = ScriptedLLM(responses={"extract": [{"events": [{
        "verb_phrase": "works at", "value": "Acme Corp", "subject": "the user",
        "subject_type": "person", "object_type": "organization",
        "polarity": "start", "modality": "happened", "occurred_at": None,
        "raw_text": "I work at Acme"}]}]})
    evs = extract_events(llm, "I work at Acme", now_iso())
    assert evs[0]["qualifier"] is None


# ---- the field survives end-to-end into the stored event ----

def test_qualifier_persisted_through_engine(tmp_path):
    extract = [{"events": [{
        "verb_phrase": "is related to", "value": "friends", "qualifier": "4 years",
        "subject": "the user", "subject_type": "person", "object_type": "person",
        "polarity": "start", "modality": "happened", "occurred_at": None,
        "raw_text": "I've known these friends for 4 years"}]}]
    branch = [{"branch": "is_related_to", "state_template": "is related to",
               "description": "relationships"}]
    mem, _, ep = _mem(tmp_path, extract, branch)
    mem.add("u", "I've known these friends for 4 years")
    stored = ep.events("u")
    assert len(stored) == 1
    assert stored[0].value == "friends"
    assert stored[0].qualifier == "4 years"


# ---- reconcile is unaffected: the qualifier is not part of identity ----

def test_differing_qualifiers_are_the_same_relationship(tmp_path):
    extract = [
        {"events": [{"verb_phrase": "is related to", "value": "friends",
                     "qualifier": "4 years", "subject": "the user",
                     "subject_type": "person", "object_type": "person",
                     "polarity": "start", "modality": "happened", "occurred_at": None,
                     "raw_text": "known these friends for 4 years"}]},
        {"events": [{"verb_phrase": "is related to", "value": "friends",
                     "qualifier": "5 years", "subject": "the user",
                     "subject_type": "person", "object_type": "person",
                     "polarity": "update", "modality": "happened", "occurred_at": None,
                     "raw_text": "known these friends for 5 years now"}]},
    ]
    branch = [{"branch": "is_related_to", "state_template": "is related to",
               "description": "relationships"}]
    mem, facts, _ = _mem(tmp_path, extract, branch)
    mem.add("u", "known these friends for 4 years")
    mem.add("u", "known these friends for 5 years now")
    # same object 'friends' -> one live fact, not two fragmented ones
    live = [f for f in facts.all("u")]
    assert len(live) == 1


# ---- step 5: the qualifier is woven back into the read/answer text ----

def _mk(value, qualifier, branch="is_related_to", occurred_at="2023-06-09",
        polarity="start"):
    return Event(id=new_id("ev"), user_id="u", entity_id="ent", branch=branch,
                 value=value, raw_text="x", polarity=polarity, recorded_at=now_iso(),
                 qualifier=qualifier, occurred_at=occurred_at)


def test_event_text_surfaces_qualifier():
    from atomir.episodic.read import _event_result
    text = _event_result(None, _mk("friends", "4 years"))["text"]
    assert text == "On 2023-06-09, the user is related to friends (4 years)"


def test_event_text_without_qualifier_is_unchanged():
    from atomir.episodic.read import _event_result
    text = _event_result(None, _mk("Acme Corp", None, branch="works_at"))["text"]
    assert text == "On 2023-06-09, the user works at Acme Corp"   # no empty parens


def test_projected_fact_text_surfaces_qualifier():
    from atomir.episodic.models import BranchRecord
    from atomir.episodic.projection import state_text
    b = BranchRecord(user_id="u", entity_id="ent", branch="is_related_to",
                     state_template="is related to", description="")
    assert state_text(b, _mk("friends", "4 years")) == \
        "The user is related to friends (4 years)"


def test_value_phrase_edge_cases():
    from atomir.episodic.models import value_phrase
    assert value_phrase("friends", "4 years") == "friends (4 years)"
    assert value_phrase("Acme", None) == "Acme"
    assert value_phrase("Acme", "") == "Acme"
    assert value_phrase("", "4 years") == "4 years"   # object empty -> no stray parens
