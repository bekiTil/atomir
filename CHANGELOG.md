# Changelog

## 0.8.2 — Temporal read quality

**Improved**
- **Temporal chain ranking**: the chain walk now ranks events by relevance to the
  sub-question (checkpoints stay pinned), so the on-topic event survives a tight
  top-k cutoff. The date stamped into each event's text preserves chronology for
  the reader.
- **Durations & ages kept intact**: extraction now records a value's time-quantity
  in a separate `qualifier` ("4 years", "since 2019", "28 years old") instead of
  letting it displace the object — "friends for 4 years" keeps `friends` as the
  value and surfaces as "friends (4 years)" on read, so "how long" questions are
  answerable. The qualifier stays out of `value`, so reconcile identity is
  unaffected; events without one render exactly as before.

## 0.8.1 — Providers + read-path performance

**New**
- **Gemini** and **Azure OpenAI** providers (LLM + embedder). Azure is
  reasoning-model aware (`reasoning_effort`, default `minimal`) and reports token
  usage including reasoning tokens; both use the shared retry with jittered
  backoff (429/5xx + `Retry-After`).
- `add(..., recorded_at=...)` sets the episodic message time so dated sessions
  anchor events correctly.

**Performance**
- The episodic read path now reuses embeddings computed at write time instead of
  re-embedding every event and branch on each query — turning an O(N) per-query
  embedding cost into O(1) after first lookup. Identical vectors; no behavior or
  quality change.

## 0.8.0 — Episodic memory (optional, experimental)

Adds an event-log layer alongside the atomic-fact store, **off by default**
(`EPISODIC_ENABLED=false`) so existing installs are unchanged.

**New**
- **Episodic layer**: messages become time-ordered *events* grouped into
  per-entity, per-verb *branches*; the fact store becomes a projection of the
  event log. A "left Beta, joined Acme" message leaves one live employer (Acme)
  with Beta in history, and the timeline keeps the transition.
- **Typed read routing** — *facts answer now, events answer when*: sub-questions
  route `current` → facts, `temporal` → a deterministic chain walk, `semantic` →
  hybrid search (with raw-episode recall).
- **General-purpose ontology**: branches emerge per-user, kept consistent by
  registry feedback; the namer emits knowledge-graph predicates (`joined` →
  `works_at`). Optional `ONTOLOGY_PACK=personal` seeds ~18 predicates.
- **New surface** (additive): `timeline()`, `forget(entity)` (cascade delete
  across facts, events, and raw episode text), `GET /timeline`, `POST /forget`,
  MCP `timeline` / `forget_about` tools, `atomir migrate --backfill`.
- **Calibration harness**: `eval/episodic/branch_microeval.py --write` tunes the
  embedder-dependent thresholds; per-embedder defaults ship in a table config
  reads.
- Per-`add` cost telemetry; `merge_entities()` maintenance primitive.
- **Gemini provider** (LLM + embedder) — `LLM_BACKEND=gemini` /
  `EMBED_BACKEND=gemini` (Google Generative Language API; no SDK).

An evaluation harness is included (`eval/episodic/`); formal benchmark results
will be published separately.

**Limits**: multi-entity graph queries out of scope; exact-alias entity
resolution (under-merges by design; `ENTITY_V2=true` opts into embedding+LLM);
JSON episodic store is dev-scale; branch-naming quality varies with model
strength (mitigated by the KG-predicate namer + calibration).

## 0.7.0
- MCP server (`atomir-mcp`): use atomir as memory in Claude Desktop / Claude Code
  / any MCP client.

## ≤ 0.6.0
- Core atomic memory: extract → reconcile on write, sub-question decomposition +
  hybrid (dense + BM25 / RRF) retrieval on read; vendor-neutral providers
  (LLM / embedder / store); LRU plan cache; multi-provider temperature.
