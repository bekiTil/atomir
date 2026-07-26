"""MCP timeline / forget_about tools. HTTP routes are smoke-tested
separately (importing atomir.api builds a store at import time)."""

from __future__ import annotations

import pytest


def test_mcp_timeline_and_forget_about(monkeypatch):
    pytest.importorskip("mcp")
    import atomir.integrations.mcp_server as m

    class Stub:
        def timeline(self, user_id, entity=None, branch=None):
            assert branch == "works_at"
            return [{"text": "The user works at Acme Corp", "occurred_at": "2025-11-01"}]

        def forget(self, user_id, entity):
            return entity == "Alex"

    monkeypatch.setattr(m, "_service", Stub())  # _memory() returns this instead of building
    out = m.timeline(entity="", branch="works_at")
    assert "Acme Corp" in out and "2025-11-01" in out
    assert "Forgotten everything about Alex" in m.forget_about("Alex")
    assert "Nothing found" in m.forget_about("Bob")


def test_mcp_advertises_new_tools(monkeypatch):
    pytest.importorskip("mcp")
    import asyncio

    import atomir.integrations.mcp_server as m
    names = {t.name for t in asyncio.run(m.mcp.list_tools())}
    assert {"remember", "recall", "list_memories", "forget", "timeline", "forget_about"} <= names
