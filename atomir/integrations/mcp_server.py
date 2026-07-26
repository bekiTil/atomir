"""atomir as an MCP server — persistent memory for Claude Desktop, Claude Code,
Cursor, or any MCP client.

Run it with the `atomir-mcp` command (installed via `pip install atomir[mcp]`).
The memory engine is configured from the environment (.env), exactly like the
rest of atomir. Single-user by default; set ATOMIR_USER to namespace it.

Add to Claude Desktop's config (claude_desktop_config.json):

    "mcpServers": {
      "atomir": {
        "command": "atomir-mcp",
        "env": {
          "LLM_BACKEND": "openai", "LLM_API_KEY": "sk-...",
          "EMBED_BACKEND": "openai", "EMBED_API_KEY": "sk-...", "EMBED_DIM": "1536",
          "STORE_BACKEND": "json", "STORE_PATH": "/absolute/path/memory.json"
        }
      }
    }
"""

from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

from atomir.assembly import build_memory_service

_USER = os.environ.get("ATOMIR_USER", "default")
_service = None  # built lazily so importing this module never needs keys


def _memory():
    global _service
    if _service is None:
        _service = build_memory_service()
    return _service


mcp = FastMCP(
    name="atomir",
    instructions=(
        "Long-term memory for the user. Call `recall` BEFORE answering anything "
        "that may depend on facts the user shared earlier (preferences, people, "
        "projects, decisions). Call `remember` AFTER learning a durable fact "
        "about the user. For 'when did X happen' or history questions, use "
        "`timeline`. For 'forget about X', use `forget_about`. Do not remember "
        "transient chit-chat."
    ),
)


@mcp.tool()
def remember(text: str) -> str:
    """Store durable facts about the user. Pass a natural-language statement;
    atomir extracts atomic facts and reconciles them (add / update / skip)."""
    facts = [f["text"] for f in _memory().add(_USER, text).get("facts", [])]
    if not facts:
        return "No new durable facts found in that."
    return "Stored:\n" + "\n".join(f"- {f}" for f in facts)


@mcp.tool()
def recall(query: str) -> str:
    """Search the user's long-term memory for facts relevant to a query."""
    res = _memory().search(_USER, query)
    facts = [r["text"] for r in res.get("results", [])]
    if not facts:
        return "No relevant memories found."
    subs = res.get("subquestions", [])
    body = "\n".join(f"- {f}" for f in facts)
    tail = f"\n\n(sub-questions asked: {', '.join(subs)})" if len(subs) > 1 else ""
    return f"Relevant memories:\n{body}{tail}"


@mcp.tool()
def list_memories() -> str:
    """List everything currently stored for the user, with ids for `forget`."""
    facts = _memory().get_all(_USER)
    if not facts:
        return "Memory is empty."
    return "\n".join(f"[{f['id']}] {f['text']}" for f in facts)


@mcp.tool()
def forget(fact_id: str) -> str:
    """Delete a single stored fact by its id (get ids from `list_memories`)."""
    ok = _memory().delete(_USER, fact_id)
    return "Forgotten." if ok else f"No fact with id {fact_id}."


@mcp.tool()
def timeline(entity: str = "", branch: str = "") -> str:
    """Show the user's time-ordered event history (when things happened / changed).
    Optionally filter to an entity (a person/org/place) and/or a relationship
    (e.g. works_at, lives_in). Requires the episodic layer to be enabled."""
    events = _memory().timeline(_USER, entity=entity or None, branch=branch or None)
    if not events:
        return "No timeline events found."
    lines = [f"- {e.get('occurred_at') or e.get('recorded_at') or '?'}: {e['text']}"
             for e in events]
    return "Timeline:\n" + "\n".join(lines)


@mcp.tool()
def forget_about(entity: str) -> str:
    """Forget EVERYTHING about a person, organization, or place by name — removes
    all related facts, events, and raw messages (cascades). Use this for
    'forget about X' requests. Requires the episodic layer to be enabled."""
    ok = _memory().forget(_USER, entity)
    return f"Forgotten everything about {entity}." if ok else f"Nothing found about {entity}."


def main() -> None:
    """Console entry point: run the server over stdio (what MCP clients launch)."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
