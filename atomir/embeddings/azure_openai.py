"""Azure OpenAI embedder — embeddings via an Azure deployment.

Same stdlib-urllib pattern; symmetric (one call for passage and query). Azure
REST: `{endpoint}/openai/deployments/{deployment}/embeddings?api-version=...`,
auth via the `api-key` header. `text-embedding-3-small` honors an explicit
`dimensions` (EMBED_DIM, default 1536). Shared retry gives 429-aware jittered
backoff for the TPM ceiling.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from atomir._http import request_bytes
from atomir.providers.embedder_base import Embedder

_USER_AGENT = "atomir/0.8"


class AzureOpenAIEmbedder(Embedder):
    def __init__(self, api_key: str, endpoint: str, deployment: str,
                 api_version: str = "2025-01-01-preview", embed_dim: int = 1536) -> None:
        if not (api_key and endpoint and deployment):
            raise ValueError(
                "AzureOpenAIEmbedder requires api_key, endpoint, and deployment "
                "(AZURE_OPENAI_KEY / _ENDPOINT / _EMBED_DEPLOYMENT)."
            )
        self.api_key = api_key
        self.embed_dim = embed_dim
        self.url = (f"{endpoint.rstrip('/')}/openai/deployments/{deployment}/embeddings"
                    f"?api-version={api_version}")

    @classmethod
    def from_config(cls, config: dict) -> "AzureOpenAIEmbedder":
        g = lambda k, e: config.get(k) or os.environ.get(e, "")
        dim = config.get("embed_dim") or os.environ.get("AZURE_OPENAI_EMBED_DIM", "1536")
        return cls(
            api_key=g("api_key", "AZURE_OPENAI_KEY"),
            endpoint=g("base_url", "AZURE_OPENAI_ENDPOINT"),
            deployment=g("model", "AZURE_OPENAI_EMBED_DEPLOYMENT"),
            api_version=g("api_version", "AZURE_OPENAI_API_VERSION") or "2025-01-01-preview",
            embed_dim=int(dim),
        )

    def _embed(self, text: str) -> list[float]:
        if not text or not text.strip():
            return [0.0] * self.embed_dim
        payload: dict = {"input": text}
        if self.embed_dim:
            payload["dimensions"] = self.embed_dim
        req = urllib.request.Request(
            self.url, data=json.dumps(payload).encode("utf-8"), method="POST",
            headers={"Content-Type": "application/json", "api-key": self.api_key,
                     "User-Agent": _USER_AGENT},
        )
        try:
            body = json.loads(request_bytes(req).decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Azure OpenAI embeddings failed: {e.code} {e.reason}") from e
        return body["data"][0]["embedding"]

    def embed_passage(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
