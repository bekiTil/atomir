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
