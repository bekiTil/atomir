# atomir

[![PyPI](https://img.shields.io/pypi/v/atomir)](https://pypi.org/project/atomir/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/atomir?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/atomir)
[![Python](https://img.shields.io/pypi/pyversions/atomir)](https://pypi.org/project/atomir/)

Atomic memory for LLM agents — **atomic on both ends**: facts are extracted and
reconciled on write; questions are decomposed into sub-questions on read.

## Why

Most memory systems store text blobs and retrieve with one fuzzy search. atomir doesn't:

- **Write** — split a message into atomic facts, then reconcile each (ADD /
  UPDATE-with-history / DELETE / NOOP). A similarity gate stops distinct facts over-merging.
- **Read** — decompose a question into sub-questions (only when useful), retrieve
  each, union the results. Surfaces facts a single-blob search misses.

**Remember _when_, not just _what_.** atomir adds an episodic layer: every
message becomes a time-ordered event on a per-relationship timeline, and the fact
store becomes a *projection* of that event log. When a state changes — a new job,
a move, a new manager — the old value isn't overwritten and lost; it stays on the
timeline. Temporal questions ("what job did I have before this one?", "who was my
manager in 2023?") are answered by **walking the timeline, not by similarity
search** — deterministic, self-hosted, no graph database. Two atomic ends (facts
on write, sub-questions on read) over a temporal spine: **facts answer _now_,
events answer _when_.**

Vendor-neutral: LLM, embedder, and store are interfaces chosen by config.
Defaults are `fake`, so it runs with no keys.

## Install

```bash
pip install atomir                          # core, offline-capable
pip install "atomir[qdrant,api]"            # Qdrant backend + HTTP API
pip install "atomir[langchain,langgraph]"   # framework integrations
```

## Quickstart

```python
from atomir.assembly import build_memory_service

mem = build_memory_service()          # backends from .env; defaults to fake (no keys)
mem.add("user123", "I'm vegetarian and my manager is Dana.")

hits = mem.search("user123", "who should I email about my project?")
print(hits["subquestions"], [r["text"] for r in hits["results"]])

mem.answer("user123", "who is my manager?")   # composed answer + the facts used
mem.get_all("user123"); mem.delete("user123", fact_id); mem.reset("user123")
```

Real providers: copy `.env.example` → `.env`, then set backends + keys.

## Providers

| Slot | Options | Config |
|---|---|---|
| LLM | `fake` `groq` `openai` `anthropic` `gemini` `azure_openai` `ollama` | `LLM_BACKEND`, `LLM_API_KEY`, `MODEL` |
| Embedder | `fake` `jina` `voyage` `openai` `gemini` `azure_openai` `ollama` | `EMBED_BACKEND`, `EMBED_API_KEY`, `EMBED_DIM` |
| Store | `json` `qdrant` | `STORE_BACKEND`, `STORE_URL` / `STORE_PATH` |

Adding a provider is one class + one registry line. `LLM_BASE_URL` /
`EMBED_BASE_URL` target self-hosted or proxy endpoints.

**Retrieval**: reads fuse dense (embedding) + lexical (BM25) rankings via RRF and
run sub-question retrievals concurrently; set `HYBRID_SEARCH=false` for dense-only.
Provider calls retry transient failures (rate limits, connection resets).

## Episodic memory (optional, experimental)

Set `EPISODIC_ENABLED=true` for an event-log layer alongside the atomic facts.
Messages become time-ordered **events** grouped into per-entity, per-verb
**branches**; the fact store becomes a projection of that log — so a "left Beta,
joined Acme" message leaves exactly one live employer (Acme) with Beta in
history, and the timeline keeps the transition.

**Arbitration — facts answer _now_, events answer _when_.** Reads route each
sub-question: `current` → facts, `temporal` → a deterministic chain walk,
`semantic` → the usual hybrid search.

```python
mem.add("u", "I left Beta in November and joined Acme Corp.")
mem.timeline("u", branch="works_at")   # ordered events (when things changed)
mem.forget("u", "Alex")                # cascade-delete everything about an entity
```

New surface (all additive): `timeline(...)`, `forget(entity)`, `GET /timeline`,
`POST /forget`, MCP `timeline` / `forget_about` tools, and `atomir migrate
--backfill --user <id>` for pre-episodic stores. **Off by default**, so upgrading
changes nothing until you opt in.

**General-purpose by default.** No ontology is assumed: branches emerge from
your own messages, kept consistent by feeding the existing registry back into the
extractor/namer/planner. The namer produces knowledge-graph predicates
(`joined` → `works_at`) so a query planner's natural hint resolves. For a known
domain, opt into a pack: `ONTOLOGY_PACK=personal` seeds ~18 predicates.

Branch matching strips entity names before embedding, uses a three-zone decision
(auto-assign / LLM judge / new), and resolves read-time hints by exact match then
relative-best. Every threshold is **embedder-dependent** — run
`python -m eval.episodic.branch_microeval --write` to calibrate
`BRANCH_MATCH_AUTO` / `BRANCH_MATCH_GRAY_LOW` / `BRANCH_RESOLVE_FLOOR` /
`BRANCH_RESOLVE_MARGIN` for your embedder.

Temporal questions are answered by walking a timeline rather than by similarity
search, so it recovers historical states that overwrite-based memory loses (a
former employer, a previous city). Deterministic, self-hosted, no graph DB. An
evaluation harness is included (`eval/episodic/`); formal benchmark results will
be published separately.

**Honest limits:** multi-entity graph queries (out of scope — Graphiti-class
systems own that); entity resolution is exact-alias by default (under-merges
rather than risk a wrong merge; `ENTITY_V2=true` adds embedding+LLM resolution);
JSON episodic store is dev-scale (append-heavy — pair with SQLite/Qdrant at
scale); and **branch-naming quality varies with model strength** — weak models
name inconsistently, mitigated by the KG-predicate namer and the calibration
harness, but a stronger model or a pack gives the most reliable chain walks.

## Agent frameworks

atomir is the memory, not the model: **recall before, remember after**. Scope
memory by `user_id` — `"user:1"` (shared), `"user:1#agent:x"` (agent-private),
`"acme|user:1"` (multi-tenant).

**LangChain** — `AtomirRetriever` is a real `BaseRetriever`:

```python
from atomir.integrations.langchain import AtomirMemory
mem = AtomirMemory(build_memory_service(), user_id="user:1")
retriever = mem.as_retriever()
```

**LangGraph** — drop-in nodes for multi-agent graphs:

```python
from atomir.integrations.langgraph import recall_node, remember_node
g.add_node("recall", recall_node(mem))       # -> state["memories"]
g.add_node("remember", remember_node(mem))   # stores state["input"]
```

Agents coordinate through shared memory (persists across runs). Store durable
findings only. Runnable examples: [`examples/`](examples/).

## Claude / MCP

`pip install "atomir[mcp]"` exposes atomir as an MCP server, giving Claude
Desktop, Claude Code, Cursor, or any MCP client persistent memory. Add to the
client's MCP config:

```json
"mcpServers": {
  "atomir": {
    "command": "atomir-mcp",
    "env": {
      "LLM_BACKEND": "openai", "LLM_API_KEY": "sk-...",
      "EMBED_BACKEND": "openai", "EMBED_API_KEY": "sk-...", "EMBED_DIM": "1536",
      "STORE_BACKEND": "json", "STORE_PATH": "/absolute/path/atomir-memory.json"
    }
  }
}
```

Claude then has four tools — `remember`, `recall`, `list_memories`, `forget` —
and carries memory across sessions. `ATOMIR_USER` namespaces the memory.

## HTTP API

Run `uvicorn atomir.api:app` (or `docker compose up`). `MemoryClient(url)` wraps
these with identical shapes.

| Method | Path | Returns |
|---|---|---|
| POST | `/memories` `{user_id, text}` | `{operations, facts}` |
| POST | `/search` `{user_id, query, k?, decompose?}` | `{subquestions, results}` |
| POST | `/answer` `{user_id, query, ...}` | `{answer, subquestions, results}` |
| GET | `/memories?user_id=` | facts |
| DELETE | `/memories/{id}?user_id=` | `{deleted, id}` |
| DELETE | `/memories?user_id=` | `{reset}` |
| GET | `/health` | `{status, store, llm, embedder}` |

## Limitations

- `RECONCILE_MIN_SIM` (default `0.5`) is embedder-dependent — re-tune with
  `eval/tune.py` when you switch embedders.
- JSON store: atomic writes, but single-process and rewrites the whole file —
  dev / small scale only; use Qdrant otherwise.
- No multi-fact transactions; a partial `add` self-heals on retry (writes are
  per-user serialized).

## License

MIT
