"""Per-add cost instrumentation.

Thin counting proxies around the LLM and Embedder. They classify each call by
the engine prompt it carries (extraction vs branch judge/name) and tally
embedding calls, so `add()` can report the REAL cost — extraction stays one
call, but branch matching adds embeddings and gray-zone judge calls. Counters
are thread-local, so concurrent adds for different users don't cross-count.
Token totals are an approximation (chars/4); exact tokens need provider metadata
the vendor-neutral interface doesn't expose.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

from atomir.providers.embedder_base import Embedder
from atomir.providers.llm_base import LLM


class _LLMCounts(threading.local):
    def __init__(self) -> None:
        self.extraction = 0
        self.judge = 0
        self.tokens = 0


class _EmbedCounts(threading.local):
    def __init__(self) -> None:
        self.embedding = 0


class CountingLLM(LLM):
    def __init__(self, inner: LLM) -> None:
        self.inner = inner
        self.c = _LLMCounts()

    def reset(self) -> None:
        self.c.extraction = self.c.judge = self.c.tokens = 0

    def chat_json(self, system: str, user: str) -> dict:
        if "extract EVENTS" in system:
            self.c.extraction += 1
        elif "branch matcher" in system or "you name a new" in system.lower():
            self.c.judge += 1
        self.c.tokens += (len(system) + len(user)) // 4
        return self.inner.chat_json(system, user)

    def chat_text(self, system: str, user: str) -> str:
        self.c.tokens += (len(system) + len(user)) // 4
        return self.inner.chat_text(system, user)


class CountingEmbedder(Embedder):
    def __init__(self, inner: Embedder) -> None:
        self.inner = inner
        self.c = _EmbedCounts()

    def reset(self) -> None:
        self.c.embedding = 0

    def embed_passage(self, text: str) -> list[float]:
        self.c.embedding += 1
        return self.inner.embed_passage(text)

    def embed_query(self, text: str) -> list[float]:
        self.c.embedding += 1
        return self.inner.embed_query(text)


class MemoizingEmbedder(Embedder):
    """Text -> vector cache (bounded LRU). The read path re-embeds the same event
    and branch texts on every query; memoizing turns that O(N) per-query API cost
    into O(1) after the first look-up — the same vectors the write path already
    computed, reused instead of recomputed. Pure performance; identical output.

    Keyed by (mode, text) so asymmetric embedders (passage vs query task hints)
    stay correct. Thread-safe: the network embed runs OUTSIDE the lock, so a rare
    race just recomputes one vector — never corrupts the cache."""

    def __init__(self, inner: Embedder, maxsize: int = 50000) -> None:
        self.inner = inner
        self.maxsize = maxsize
        self._cache: "OrderedDict[tuple[str, str], list[float]]" = OrderedDict()
        self._lock = threading.Lock()

    def reset(self) -> None:  # delegate per-add counter reset; keep the cache warm
        if hasattr(self.inner, "reset"):
            self.inner.reset()

    @property
    def embed_dim(self):
        return getattr(self.inner, "embed_dim", None)

    def _memo(self, mode: str, text: str, fn) -> list[float]:
        key = (mode, text)
        with self._lock:
            v = self._cache.get(key)
            if v is not None:
                self._cache.move_to_end(key)
                return v
        v = fn(text)  # embed outside the lock (network I/O)
        with self._lock:
            self._cache[key] = v
            self._cache.move_to_end(key)
            while len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)
        return v

    def embed_passage(self, text: str) -> list[float]:
        return self._memo("p", text, self.inner.embed_passage)

    def embed_query(self, text: str) -> list[float]:
        return self._memo("q", text, self.inner.embed_query)
