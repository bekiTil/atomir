"""Negation render guard."""

from __future__ import annotations

from atomir.episodic.models import BranchRecord, Event, new_id, now_iso
from atomir.episodic.projection import _clean_template, state_text


def _end_event(value="Beta"):
    return Event(id=new_id("ev"), user_id="u", entity_id="ent", branch="works_at",
                 value=value, raw_text="x", polarity="end", recorded_at=now_iso())


def _branch(template):
    return BranchRecord(branch="works_at", user_id="u", entity_id="ent",
                        description="emp", state_template=template, object_type="organization")


def test_clean_template_strips_leading_end_verbs():
    assert _clean_template("left works at") == "works at"
    assert _clean_template("quit works at") == "works at"
    assert _clean_template("works at") == "works at"


def test_end_negation_never_doubles_the_verb():
    # The exact regression: a template that absorbed the verb "left".
    assert state_text(_branch("left works at"), _end_event("Beta")) == \
        "The user no longer works at Beta"
    # A clean template stays clean.
    assert state_text(_branch("works at"), _end_event("Beta Inc")) == \
        "The user no longer works at Beta Inc"


def test_start_state_is_clean_too():
    ev = Event(id=new_id("ev"), user_id="u", entity_id="ent", branch="works_at",
               value="Acme", raw_text="x", polarity="start", recorded_at=now_iso())
    assert state_text(_branch("works at"), ev) == "The user works at Acme"
