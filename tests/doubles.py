"""Deterministic test doubles for the characterization suite.

The stock `FakeLLM`/`FakeEmbedder` can't exercise the reconcile UPDATE/DELETE
paths (the similarity gate short-circuits to ADD, and one canned dict can't serve
extract + reconcile + plan differently). These doubles give per-prompt scripting
and exact cosine control so current behavior can be pinned precisely.
"""

from __future__ import annotations

import re

from atomir.embeddings.fake import FakeEmbedder
from atomir.llm.fake import FakeLLM
from atomir.memory import MemoryService
from atomir.providers.embedder_base import Embedder
from atomir.providers.llm_base import LLM

# Shipped production default (env RECONCILE_MIN_SIM default is 0.5, applied in
# assembly). Match it so characterization mirrors real usage, not the bare
# MemoryService default of 0.6.
PROD_MIN_SIM = 0.5


class ScriptedLLM(LLM):
    """LLM whose response depends on which engine prompt it sees.

    `responses` maps a mode ("extract"/"reconcile"/"plan"/"text") -> FIFO list.
    Each item is a dict/str, or a callable(system, user) -> dict/str (so a
    reconcile response can read neighbour ids out of the prompt). Exhausted
    queues fall back to a safe default. Per-mode call counts are recorded.
    """

    def __init__(self, responses: dict | None = None) -> None:
        self.responses = {k: list(v) for k, v in (responses or {}).items()}
        self.calls: dict[str, int] = {
            "extract": 0, "reconcile": 0, "plan": 0, "text": 0,
            "branch_judge": 0, "branch_name": 0, "entity_judge": 0, "summary": 0,
        }

    @staticmethod
    def _mode(system: str) -> str:
        if "query planner" in system:
            return "plan"
        if "choose exactly ONE action" in system:
            return "reconcile"
        if "entity resolver" in system:
            return "entity_judge"
        if "for a checkpoint" in system:
            return "summary"
        if "branch matcher" in system:
            return "branch_judge"
        if "you name a new" in system.lower():
            return "branch_name"
        if "classify a personal-memory predicate" in system:
            return "cardinality"
        return "extract"

    def _pop(self, mode: str, system: str, user: str, default):
        self.calls[mode] += 1
        queue = self.responses.get(mode, [])
        item = queue.pop(0) if queue else default
        return item(system, user) if callable(item) else item

    def chat_json(self, system: str, user: str) -> dict:
        mode = self._mode(system)
        defaults = {
            "plan": {"decompose": False, "subquestions": [user]},
            "reconcile": {"decision": "ADD", "target_id": None, "reason": "default"},
            "branch_judge": {"branch": "NEW"},
            "branch_name": {},  # empty -> matcher's deterministic fallback
            "entity_judge": {"same": False},
            "summary": {"summary": ""},
            "extract": {"facts": []},
            "cardinality": {"cardinality": "single"},
        }
        return self._pop(mode, system, user, defaults[mode])

    def chat_text(self, system: str, user: str) -> str:
        return self._pop("text", system, user, "")


def first_neighbor_id(reconcile_prompt: str) -> str | None:
    """Parse the first `id=...` from a reconcile prompt's neighbour listing."""
    m = re.search(r"id=(\S+)", reconcile_prompt)
    return m.group(1) if m else None


class StubEmbedder(Embedder):
    """Exact-control embedder: texts in the same declared synonym group share a
    one-hot vector (cosine 1.0); every other distinct text gets its own
    orthogonal one-hot (cosine 0.0). Stateful within one instance; deterministic
    per text. Reuse ONE instance across add + search so vectors stay consistent.
    """

    def __init__(self, synonyms: list[list[str]] | None = None, dim: int = 256) -> None:
        self.dim = dim
        self._group: dict[str, int] = {}
        self._own: dict[str, int] = {}
        self._next = 0
        for group in synonyms or []:
            gi = self._alloc()
            for t in group:
                self._group[t] = gi

    def _alloc(self) -> int:
        i = self._next
        self._next += 1
        return i

    def _vec(self, text: str) -> list[float]:
        if text in self._group:
            idx = self._group[text]
        else:
            idx = self._own.setdefault(text, self._alloc())
        v = [0.0] * self.dim
        v[idx % self.dim] = 1.0
        return v

    def embed_passage(self, text: str) -> list[float]:
        return self._vec(text)

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def make_service(store, llm=None, embedder=None, **kw) -> MemoryService:
    """Build a MemoryService mirroring assembly's production defaults."""
    kw.setdefault("reconcile_min_sim", PROD_MIN_SIM)
    return MemoryService(store, llm or FakeLLM(), embedder or FakeEmbedder(), **kw)
