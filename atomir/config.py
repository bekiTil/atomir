"""Centralized, vendor-neutral configuration.

12-factor (factor III): all config that varies between deploys — keys, model
names, URLs, paths — is read from the environment, never hardcoded. A local
`.env` file is loaded for convenience in development.

The design exposes THREE independent, swappable provider slots — `llm`,
`embedder`, and `vector_store` — each as a mem0-style `{provider, config}`
block. No vendor is privileged: whichever providers the environment selects are
the ones used. The first-run example happens to be Groq + Jina + Qdrant, but
each is just one value of an interface that gets formalized in later steps.

Defaults favor running with NO external keys: both the LLM and embedder default
to the `fake` backend so the system is testable offline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Load .env into os.environ if present (no error if the file is missing).
load_dotenv()


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _threshold(env_name: str, key: str) -> float:
    """A branch/resolution threshold: env override wins, else the per-embedder
    calibrated default from the thresholds table (see eval/episodic/branch_microeval)."""
    override = _env(env_name)
    if override:
        return float(override)
    from atomir.episodic.thresholds import get_thresholds
    return get_thresholds(_env("EMBED_BACKEND", "fake"))[key]


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of configuration read from the environment."""

    # --- LLM slot ---------------------------------------------------------
    # Field names stay vendor-neutral: `llm_backend` selects the impl, and the
    # key/model are read into generic fields. `groq` is just the first example.
    llm_backend: str = field(default_factory=lambda: _env("LLM_BACKEND", "fake"))
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY"))
    model: str = field(default_factory=lambda: _env("MODEL", "llama-3.3-70b-versatile"))
    # Optional override of the LLM endpoint (self-host, proxy, local Ollama).
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL"))
    # Unset = the provider's own default. 0 = greedy/deterministic decoding.
    llm_temperature: str = field(default_factory=lambda: _env("LLM_TEMPERATURE"))
    # Best-effort reproducibility for providers that honor it (e.g. OpenAI).
    llm_seed: str = field(default_factory=lambda: _env("LLM_SEED"))

    # --- Embedder slot ----------------------------------------------------
    # Same neutrality: `jina` is the first example, but nothing here names it.
    embed_backend: str = field(default_factory=lambda: _env("EMBED_BACKEND", "fake"))
    embed_api_key: str = field(default_factory=lambda: _env("EMBED_API_KEY"))
    # Dimension travels WITH the embedder choice (Jina v3 = 1024).
    embed_dim: int = field(default_factory=lambda: int(_env("EMBED_DIM", "1024")))
    embed_base_url: str = field(default_factory=lambda: _env("EMBED_BASE_URL"))

    # --- Retrieval tuning -------------------------------------------------
    # Hybrid read: fuse dense (embedding) + lexical (BM25) rankings with RRF.
    # Semantic-only search buries facts that hinge on rare terms (names, titles).
    hybrid_search: bool = field(
        default_factory=lambda: _env("HYBRID_SEARCH", "true").lower() != "false"
    )
    # Cache query decompositions (LRU) so repeat queries skip the planner call.
    plan_cache: bool = field(
        default_factory=lambda: _env("PLAN_CACHE", "true").lower() != "false"
    )

    # --- Reconciler tuning ------------------------------------------------
    # Write-side similarity gate (DECISION #1a): a candidate whose nearest
    # existing fact scores below this is ADDed directly, without asking the LLM.
    # Tuned via eval/tune.py: sits between measured unrelated (~0.45) and
    # same-attribute (~0.60) similarity on Jina; re-tune per embedder.
    reconcile_min_sim: float = field(
        default_factory=lambda: float(_env("RECONCILE_MIN_SIM", "0.5"))
    )

    # --- Episodic memory (additive; OFF by default) -----------------------
    # The event-log / timeline layer. Default OFF so upgrading never changes an
    # existing install's behavior; turn it on to opt in.
    # With it off, add/search/answer behave exactly as the current release.
    episodic_enabled: bool = field(
        default_factory=lambda: _env("EPISODIC_ENABLED", "false").lower() == "true"
    )
    # Three-zone branch matcher. AUTO is per-embedder; the gray
    # band [GRAY_LOW, AUTO) is routed to the LLM judge. Tunable via eval/episodic.
    branch_match_auto: float = field(
        default_factory=lambda: _threshold("BRANCH_MATCH_AUTO", "auto"))
    branch_match_gray_low: float = field(
        default_factory=lambda: _threshold("BRANCH_MATCH_GRAY_LOW", "gray_low"))
    # Entity resolution v2: when on, an unknown mention that matches no
    # alias exactly is compared to existing entities by embedding, and an LLM
    # confirms an ambiguous match (else a new entity). Off keeps cheap v1
    # exact-alias matching (under-merges rather than risk a wrong merge).
    entity_v2: bool = field(
        default_factory=lambda: _env("ENTITY_V2", "false").lower() == "true"
    )
    entity_match_min: float = field(
        default_factory=lambda: float(_env("ENTITY_MATCH_MIN", "0.60"))
    )
    # Summarize a branch's oldest segment once it exceeds this many nodes.
    checkpoint_every: int = field(
        default_factory=lambda: int(_env("CHECKPOINT_EVERY", "50"))
    )
    # Optional ontology pack to pre-seed a user's branch registry for a known
    # domain (e.g. "personal"). Default empty = general-purpose: seed nothing,
    # let the ontology emerge per-user (kept consistent by registry feedback).
    ontology_pack: str = field(default_factory=lambda: _env("ONTOLOGY_PACK"))
    # Read-side branch_hint resolution (relative-best): resolve to the top
    # branch only when it clears FLOOR and beats the runner-up by MARGIN.
    branch_resolve_floor: float = field(
        default_factory=lambda: _threshold("BRANCH_RESOLVE_FLOOR", "floor"))
    branch_resolve_margin: float = field(
        default_factory=lambda: _threshold("BRANCH_RESOLVE_MARGIN", "margin"))
    # Empty = use STORE_BACKEND for episodic data too; or "sqlite".
    episodic_store: str = field(default_factory=lambda: _env("EPISODIC_STORE"))
    # Checkpoint summary source. "raw" (default, shipped 0.8.4): summarise
    # from event value strings. "structured": summarise from
    # (value, qualifier, polarity, occurred_at) tuples so the resulting text
    # carries the dates and qualifiers of the absorbed events and contains no
    # information not already stored structurally.
    checkpoint_source: str = field(
        default_factory=lambda: _env("CHECKPOINT_SOURCE", "raw"))
    # Temporal-route checkpoint expansion. "off" (default): temporal reads
    # return checkpoints plus non-absorbed events. "on": each returned
    # checkpoint is expanded into its absorbed_event_ids for ranking. Fact
    # reads unaffected. Requires absorbed_event_ids on checkpoint records.
    temporal_expand_checkpoints: str = field(
        default_factory=lambda: _env("ATOMIR_TEMPORAL_EXPAND_CHECKPOINTS", "off"))
    # Fact projection mode. "off" (default): single fact per branch, updated
    # in place. "on": each write appends a new fact and demotes the prior one
    # via metadata.is_current=False; branch keeps fact_history_ids.
    append_only_facts: str = field(
        default_factory=lambda: _env("ATOMIR_APPEND_ONLY_FACTS", "off"))

    # --- Vector store slot ------------------------------------------------
    # Selected by `store_backend` like the other two slots (no hardcoded vendor).
    store_backend: str = field(default_factory=lambda: _env("STORE_BACKEND", "qdrant"))
    collection: str = field(default_factory=lambda: _env("COLLECTION", "atomir_memories"))
    # `url`/`path` are generic connection params (Qdrant uses both today).
    store_url: str = field(default_factory=lambda: _env("STORE_URL"))
    store_path: str = field(default_factory=lambda: _env("STORE_PATH", "./qdrant_data"))

    # --- Vendor-neutral provider blocks (mem0-style {provider, config}) ---
    # These are how the engine will eventually select implementations without
    # ever naming a vendor. The factories that consume them arrive in Step 5.5.

    @property
    def llm(self) -> dict:
        return {
            "provider": self.llm_backend,
            "config": {
                "api_key": self.llm_api_key,
                "model": self.model,
                "base_url": self.llm_base_url,
                "temperature": (
                    float(self.llm_temperature) if self.llm_temperature else None
                ),
                "seed": (int(self.llm_seed) if self.llm_seed else None),
            },
        }

    @property
    def embedder(self) -> dict:
        return {
            "provider": self.embed_backend,
            "config": {
                "api_key": self.embed_api_key,
                "embed_dim": self.embed_dim,
                "base_url": self.embed_base_url,
            },
        }

    @property
    def vector_store(self) -> dict:
        return {
            "provider": self.store_backend,
            "config": {
                "url": self.store_url,
                "path": self.store_path,
                "collection": self.collection,
                "embed_dim": self.embed_dim,
            },
        }


# A single shared instance importers use: `from atomir.config import settings`.
settings = Settings()


def make_qdrant_client():
    """Build a QdrantClient from settings: server mode if a URL is set, else
    embedded/local at the configured path (zero-ops).

    The qdrant SDK is imported lazily so merely importing this config module
    never requires the optional provider dependency (offline/fake path stays
    dependency-free). Returns a `qdrant_client.QdrantClient`.
    """
    from qdrant_client import QdrantClient

    if settings.store_url:
        return QdrantClient(url=settings.store_url)
    return QdrantClient(path=settings.store_path)
