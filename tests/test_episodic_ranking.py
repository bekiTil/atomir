"""Temporal ranking: the event most relevant to the question ranks first, so it
survives a tight top-k cutoff (the top_10 lift)."""

from __future__ import annotations

from atomir.episodic.models import Event, new_id, now_iso
from atomir.episodic.read import _event_result, _rank_events
from doubles import StubEmbedder


def _ev(value, date="2023-05-07"):
    return Event(id=new_id("ev"), user_id="u", entity_id="ent", branch="attends",
                 value=value, raw_text="x", polarity="start", recorded_at=now_iso(),
                 occurred_at=date)


def test_relevance_ranking_surfaces_matching_event_first():
    events = [_ev("book club"), _ev("LGBTQ support group"), _ev("pottery class"),
              _ev("guitar lessons")]
    question = "when did the user go to the support group"
    match_text = _event_result(None, events[1])["text"]   # the relevant event's text
    # StubEmbedder: the question and the matching event embed identically (cosine
    # 1.0); every other event is orthogonal (0.0).
    emb = StubEmbedder(synonyms=[[question, match_text]])

    ranked = _rank_events(emb, events, question, k=4, order="relevance")
    assert ranked[0].value == "LGBTQ support group"        # most relevant is #1

    # 'time' order still sorts chronologically (unchanged default).
    chrono = _rank_events(emb, events, question, k=4, order="time")
    assert [e.value for e in chrono][0] in {"book club", "LGBTQ support group",
                                            "pottery class", "guitar lessons"}
