"""Gemini embedder — Google Generative Language API (embedContent).

`gemini-embedding-001` is asymmetric: passages and queries use different task
types (RETRIEVAL_DOCUMENT vs RETRIEVAL_QUERY), like Jina. It supports an
`outputDimensionality` (768 / 1536 / 3072), so EMBED_DIM is honored directly.
Stdlib urllib + shared retry; no SDK. Key in the `x-goog-api-key` header.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from atomir._http import request_bytes
from atomir.providers.embedder_base import Embedder

_DEFAULT_BASE = "https://generativelanguage.googleapis.com/v1beta"
_USER_AGENT = "atomir/0.8"


class GeminiEmbedder(Embedder):
    def __init__(self, api_key: str, embed_dim: int = 768,
                 model: str = "gemini-embedding-001", base_url: str = "") -> None:
        if not api_key:
            raise ValueError(
                "GeminiEmbedder requires an API key. Set EMBED_API_KEY, or use "
                "EMBED_BACKEND=fake to run offline."
            )
        self.api_key = api_key
        self.embed_dim = embed_dim
        self.model = model
        self.base = base_url.rstrip("/") if base_url else _DEFAULT_BASE

    @classmethod
    def from_config(cls, config: dict) -> "GeminiEmbedder":
        return cls(
            api_key=config.get("api_key", ""),
            embed_dim=config.get("embed_dim", 768),
            model=config.get("model", "gemini-embedding-001"),
            base_url=config.get("base_url", ""),
        )

    def _embed(self, text: str, task: str) -> list[float]:
        if not text or not text.strip():
            return [0.0] * self.embed_dim
        body: dict = {
            "model": f"models/{self.model}",
            "content": {"parts": [{"text": text}]},
            "taskType": task,
        }
        if self.embed_dim:  # gemini-embedding-001 honors an explicit dimensionality
            body["outputDimensionality"] = self.embed_dim
        req = urllib.request.Request(
            f"{self.base}/models/{self.model}:embedContent",
            data=json.dumps(body).encode("utf-8"), method="POST",
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
                "User-Agent": _USER_AGENT,
            },
        )
        try:
            payload = json.loads(request_bytes(req).decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Gemini embedding failed: {e.code} {e.reason}") from e
        return payload["embedding"]["values"]

    def embed_passage(self, text: str) -> list[float]:
        return self._embed(text, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text, "RETRIEVAL_QUERY")
