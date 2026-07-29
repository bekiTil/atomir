"""Azure OpenAI provider: registration, config, request shaping, usage
accounting (incl. reasoning_tokens), and the 429-retry path — all offline."""

from __future__ import annotations

import email.message
import json
import urllib.error
import urllib.request

import pytest

from atomir.embeddings.azure_openai import AzureOpenAIEmbedder
from atomir.llm.azure_openai import AzureOpenAILLM
from atomir.providers.factory import EmbedderFactory, LLMFactory


def test_factory_registers_azure():
    assert "azure_openai" in LLMFactory._registry()
    assert "azure_openai" in EmbedderFactory._registry()


def test_llm_from_config_env_fallback(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_KEY", "k")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    llm = AzureOpenAILLM.from_config({})
    assert "deployments/gpt-5-mini/chat/completions" in llm.url
    assert "api-version=2025-01-01-preview" in llm.url
    assert llm.reasoning_effort == "minimal"


def _canned_chat(monkeypatch, capture):
    def fake_request_bytes(req, **kw):
        capture["req"] = req
        return json.dumps({
            "choices": [{"message": {"content": '{"ok": true}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                      "completion_tokens_details": {"reasoning_tokens": 3}},
        }).encode()
    monkeypatch.setattr("atomir.llm.azure_openai.request_bytes", fake_request_bytes)


def test_llm_request_shaping_and_usage(monkeypatch):
    cap: dict = {}
    _canned_chat(monkeypatch, cap)
    llm = AzureOpenAILLM(api_key="k", endpoint="https://x.openai.azure.com",
                         deployment="gpt-5-mini", reasoning_effort="minimal")
    out = llm.chat_json("sys", "user")
    assert out == {"ok": True}

    body = json.loads(cap["req"].data.decode())
    assert body["reasoning_effort"] == "minimal"            # hidden-thinking budget capped
    assert body["response_format"] == {"type": "json_object"}
    assert "temperature" not in body                        # reasoning model: not sent
    assert cap["req"].get_header("Api-key") == "k"          # api-key header auth
    # usage accounting captures reasoning_tokens
    assert llm.usage == {"prompt": 10, "completion": 5, "reasoning": 3, "calls": 1}


def test_embedder_request_shaping(monkeypatch):
    cap: dict = {}
    def fake_request_bytes(req, **kw):
        cap["req"] = req
        return json.dumps({"data": [{"embedding": [0.1] * 1536}]}).encode()
    monkeypatch.setattr("atomir.embeddings.azure_openai.request_bytes", fake_request_bytes)
    emb = AzureOpenAIEmbedder(api_key="k", endpoint="https://x.openai.azure.com",
                              deployment="text-embedding-3-small", embed_dim=1536)
    vec = emb.embed_query("hello")
    assert len(vec) == 1536
    body = json.loads(cap["req"].data.decode())
    assert body["dimensions"] == 1536 and body["input"] == "hello"
    assert "deployments/text-embedding-3-small/embeddings" in cap["req"].full_url


def test_http_retries_on_429_with_jitter(monkeypatch):
    from atomir import _http
    calls = {"n": 0}

    class FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"ok":1}'

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            hdrs = email.message.Message()
            hdrs["Retry-After"] = "0"
            raise urllib.error.HTTPError(req.full_url, 429, "rate limited", hdrs, None)
        return FakeResp()

    monkeypatch.setattr(_http.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(_http.time, "sleep", lambda *a: None)
    body = _http.request_bytes(
        urllib.request.Request("http://x", data=b"{}", method="POST"))
    assert body == b'{"ok":1}' and calls["n"] == 2            # retried once, then succeeded
