"""Non-destructive migration/backfill for legacy stores."""

from __future__ import annotations

from atomir.embeddings.fake import FakeEmbedder
from atomir.episodic.engine import EpisodicMemory
from atomir.episodic.json_store import JsonEpisodicStore
from atomir.llm.fake import FakeLLM
from atomir.memory import MemoryService
from atomir.stores.json_store import JsonMemoryStore


def _legacy(tmp_path):
    """A facts store with pre-episodic data and an episodic engine over it."""
    facts = JsonMemoryStore(path=str(tmp_path / "facts.json"))
    facts.add("u", "The user works at Acme Corp", [0.0] * 8)   # legacy fact, no events
    ep = JsonEpisodicStore(path=str(tmp_path / "ep.json"))
    engine = EpisodicMemory(facts, ep, FakeLLM(), FakeEmbedder(),
                            branch_auto=0.8, branch_gray_low=0.4)
    return facts, ep, engine


def test_backfill_synthesizes_event_per_fact_idempotently(tmp_path):
    facts, ep, engine = _legacy(tmp_path)
    assert engine.backfill("u") == 1

    events = ep.events("u")
    assert len(events) == 1
    e = events[0]
    assert e.branch == "legacy" and e.polarity == "update" and e.occurred_at is None
    assert e.fact_id == facts.all("u")[0]["id"]          # linked to the legacy fact
    # Timeline now populated; facts untouched.
    assert engine.timeline("u")[0]["value"] == "The user works at Acme Corp"
    assert len(facts.all("u")) == 1

    assert engine.backfill("u") == 0                      # idempotent second run


def test_cli_migrate_backfill(tmp_path, monkeypatch, capsys):
    facts, ep, engine = _legacy(tmp_path)
    svc = MemoryService(facts, FakeLLM(), FakeEmbedder(), episodic=engine)
    import atomir.assembly as assembly
    monkeypatch.setattr(assembly, "build_memory_service", lambda *a, **k: svc)

    from atomir.cli import main
    assert main(["migrate", "--backfill", "--user", "u"]) == 0
    assert "Backfilled 1 event" in capsys.readouterr().out
    assert len(ep.events("u")) == 1
