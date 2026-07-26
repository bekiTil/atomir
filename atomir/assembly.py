"""Assembly: build a `MemoryService` from config.

This is the OUTERMOST wiring layer — the one place allowed to name concrete
backends. It selects the LLM and embedder through their factories and the store
through a small backend switch, then injects all three into the (vendor-neutral)
`MemoryService`. Concrete backends are imported lazily so building a fake/JSON
service never requires an unused provider SDK.
"""

from __future__ import annotations

from atomir.config import Settings
from atomir.config import make_qdrant_client
from atomir.config import settings as default_settings
from atomir.memory import MemoryService
from atomir.providers import EmbedderFactory, LLMFactory
from atomir.store_base import MemoryStore


def _build_store(s: Settings) -> MemoryStore:
    provider = s.store_backend
    if provider == "qdrant":
        from atomir.stores.qdrant_store import QdrantMemoryStore

        return QdrantMemoryStore(make_qdrant_client(), s.collection, s.embed_dim)
    if provider == "json":
        from atomir.stores.json_store import JsonMemoryStore

        path = s.store_path if s.store_path.endswith(".json") else "./atomir_store.json"
        return JsonMemoryStore(path=path)
    raise ValueError(
        f"Unknown store backend {provider!r}. Valid backends: ['qdrant', 'json']"
    )


def _episodic_path(s: Settings, ext: str) -> str:
    """Where the episodic store lives: next to the facts store when that is a
    JSON file, else a default. `ext` is '.episodic.json' or '.episodic.db'."""
    p = s.store_path
    if p.endswith(".json"):
        return p[:-len(".json")] + ext
    return "./atomir_episodic" + ext


def _build_episodic(s: Settings, facts, llm, embedder):
    from atomir.episodic.engine import EpisodicMemory

    if s.episodic_store == "sqlite":
        from atomir.episodic.sqlite_store import SqliteEpisodicStore

        ep_store = SqliteEpisodicStore(path=_episodic_path(s, ".episodic.db"))
    else:
        from atomir.episodic.json_store import JsonEpisodicStore

        ep_store = JsonEpisodicStore(path=_episodic_path(s, ".episodic.json"))
    return EpisodicMemory(facts, ep_store, llm, embedder,
                          branch_auto=s.branch_match_auto,
                          branch_gray_low=s.branch_match_gray_low,
                          entity_v2=s.entity_v2, entity_match_min=s.entity_match_min,
                          checkpoint_every=s.checkpoint_every, ontology_pack=s.ontology_pack,
                          resolve_floor=s.branch_resolve_floor,
                          resolve_margin=s.branch_resolve_margin)


def build_memory_service(s: Settings = default_settings) -> MemoryService:
    """Construct a MemoryService with providers/backend selected purely by config."""
    llm = LLMFactory.create(s.llm)
    embedder = EmbedderFactory.create(s.embedder)
    store = _build_store(s)
    # Episodic layer is built (and delegated to) only when the flag is on.
    episodic = _build_episodic(s, store, llm, embedder) if s.episodic_enabled else None
    return MemoryService(store, llm, embedder, reconcile_min_sim=s.reconcile_min_sim,
                         hybrid_search=s.hybrid_search, cache_plans=s.plan_cache,
                         episodic=episodic)
