"""Anti-loss (0.8.3): for a direct temporal lookup, the source clause is surfaced
when the (verb, value) projection dropped content — but ONLY then. Multi-part
queries keep the clean projected text, so the extra clause doesn't add noise."""

from __future__ import annotations

from atomir.episodic.models import Event, new_id, now_iso
from atomir.episodic.read import _adds_signal, _event_result


def _ev():
    # projection kept only "hand-painted bowl"; the clause carries the birthday.
    return Event(id=new_id("ev"), user_id="u", entity_id="ent", branch="received",
                 value="hand-painted bowl", polarity="start", recorded_at=now_iso(),
                 occurred_at="2016-07-31",
                 raw_text="my friend made me this for my 18th birthday ten years ago")


def test_temporal_lookup_surfaces_the_lost_clause():
    text = _event_result(None, _ev(), append_raw=True)["text"]
    assert "18th birthday" in text          # the dropped detail is now findable
    assert text.startswith("On 2016-07-31,")


def test_non_temporal_query_keeps_clean_text():
    text = _event_result(None, _ev(), append_raw=False)["text"]
    assert "18th birthday" not in text       # scoped OFF -> no clause appended
    assert text == "On 2016-07-31, the user received hand-painted bowl"


def test_default_is_clean_no_append():
    # default append_raw=False: nothing changes vs the base projection.
    assert "18th birthday" not in _event_result(None, _ev())["text"]


def test_adds_signal_only_when_clause_adds_content():
    assert _adds_signal("made for my 18th birthday ten years ago", "the user received bowl")
    assert not _adds_signal("a bowl", "the user received a bowl")   # pure echo
