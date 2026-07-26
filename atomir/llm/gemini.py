"""Gemini LLM — Google Generative Language API (generateContent).

Stdlib urllib + shared retry + JSON mode + lenient parse; no SDK. The API key
goes in the `x-goog-api-key` header (never the URL). `base_url` lets you target a
proxy or a pinned API version. NOT live-tested here — verify with a real key.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from atomir._http import request_bytes
from atomir.llm.parsing import extract_json, judge_with

_DEFAULT_BASE = "https://generativelanguage.googleapis.com/v1beta"
_USER_AGENT = "atomir/0.8"


class GeminiLLM:
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash",
                 base_url: str = "", temperature: float | None = None) -> None:
        if not api_key:
            raise ValueError(
                "GeminiLLM requires an API key. Set LLM_API_KEY, or use "
                "LLM_BACKEND=fake to run offline."
            )
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.base = base_url.rstrip("/") if base_url else _DEFAULT_BASE

    @classmethod
    def from_config(cls, config: dict) -> "GeminiLLM":
        return cls(
            api_key=config.get("api_key", ""),
            model=config.get("model", "gemini-2.0-flash"),
            base_url=config.get("base_url", ""),
            temperature=config.get("temperature"),
        )

    def _chat(self, system: str, user: str, json_mode: bool) -> str:
        gen: dict = {}
        if self.temperature is not None:
            gen["temperature"] = self.temperature
        if json_mode:
            gen["responseMimeType"] = "application/json"
        body: dict = {
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "systemInstruction": {"parts": [{"text": system}]},
        }
        if gen:
            body["generationConfig"] = gen
        req = urllib.request.Request(
            f"{self.base}/models/{self.model}:generateContent",
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
            detail = e.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"Gemini request failed: {e.code} {e.reason} — {detail}") from e
        parts = payload["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)

    def chat_text(self, system: str, user: str) -> str:
        return self._chat(system, user, json_mode=False)

    def chat_json(self, system: str, user: str) -> dict:
        return extract_json(self._chat(system, user, json_mode=True))

    def judge(self, rubric: str, content: str) -> tuple[bool, str]:
        return judge_with(self.chat_json, rubric, content)
