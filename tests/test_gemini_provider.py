"""Gemini LLM + embedder: factory wiring and request/response shaping.

No live key — the HTTP layer is monkeypatched, so this checks the request is
built correctly (key in the header not the URL, JSON mode, task types) and the
responses parse.
"""

from __future__ import annotations

import json

import atomir.embeddings.gemini as gemb
import atomir.llm.gemini as gllm
from atomir.embeddings.gemini import GeminiEmbedder
from atomir.llm.gemini import GeminiLLM
from atomir.providers.factory import EmbedderFactory, LLMFactory


def test_factory_registers_gemini():
    llm = LLMFactory.create({"provider": "gemini", "config": {"api_key": "k"}})
    assert isinstance(llm, GeminiLLM)
    emb = EmbedderFactory.create({"provider": "gemini", "config": {"api_key": "k"}})
    assert isinstance(emb, GeminiEmbedder)


def test_gemini_llm_chat_json_shaping(monkeypatch):
    seen = {}

    def fake(req, **kw):
        seen["url"] = req.full_url
        seen["headers"] = {k.lower(): v for k, v in req.headers.items()}
        seen["body"] = json.loads(req.data.decode())
        return json.dumps({"candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}]}).encode()

    monkeypatch.setattr(gllm, "request_bytes", fake)
    out = GeminiLLM(api_key="secret", model="gemini-2.0-flash").chat_json("SYS", "USER")
    assert out == {"ok": True}
    assert "secret" not in seen["url"]                       # key never in URL
    assert seen["headers"]["x-goog-api-key"] == "secret"     # key in header
    assert seen["body"]["systemInstruction"]["parts"][0]["text"] == "SYS"
    assert seen["body"]["generationConfig"]["responseMimeType"] == "application/json"


def test_gemini_embedder_task_types(monkeypatch):
    tasks = []

    def fake(req, **kw):
        tasks.append(json.loads(req.data.decode())["taskType"])
        return json.dumps({"embedding": {"values": [0.1, 0.2, 0.3]}}).encode()

    monkeypatch.setattr(gemb, "request_bytes", fake)
    e = GeminiEmbedder(api_key="k", embed_dim=768)
    assert e.embed_passage("hi") == [0.1, 0.2, 0.3]
    assert e.embed_query("hi") == [0.1, 0.2, 0.3]
    assert tasks == ["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]   # asymmetric
    assert e.embed_passage("   ") == [0.0] * 768               # blank -> zero vec, no call


def test_gemini_requires_key():
    import pytest
    with pytest.raises(ValueError):
        GeminiLLM(api_key="")
    with pytest.raises(ValueError):
        GeminiEmbedder(api_key="")
