"""OpenAI LLM — OpenAI-compatible chat completions (same shape as Groq).

Stdlib urllib + shared retry + JSON mode + lenient parse. NOT live-tested here —
verify with a real key. `base_url` also lets you target any OpenAI-compatible
endpoint (Azure OpenAI, local servers, proxies).
"""

from __future__ import annotations

import json
import warnings
import urllib.error
import urllib.request

from atomir._http import request_bytes
from atomir.llm.parsing import extract_json, judge_with

_DEFAULT_BASE = "https://api.openai.com/v1"
_USER_AGENT = "atomir/0.2"


class OpenAILLM:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", base_url: str = "",
                 temperature: float | None = None, seed: int | None = None) -> None:
        if not api_key:
            raise ValueError(
                "OpenAILLM requires an API key. Set LLM_API_KEY, or use "
                "LLM_BACKEND=fake to run offline."
            )
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.seed = seed  # best-effort reproducibility (OpenAI honors it)
        self.url = (base_url.rstrip("/") if base_url else _DEFAULT_BASE) + "/chat/completions"

    @classmethod
    def from_config(cls, config: dict) -> "OpenAILLM":
        return cls(
            api_key=config.get("api_key", ""),
            model=config.get("model", "gpt-4o-mini"),
            base_url=config.get("base_url", ""),
            temperature=config.get("temperature"),
            seed=config.get("seed"),
        )

    def _chat(self, system: str, user: str, json_mode: bool) -> str:
        body: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.seed is not None:
            body["seed"] = self.seed
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(
            self.url, data=json.dumps(body).encode("utf-8"), method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": _USER_AGENT,
            },
        )
        try:
            payload = json.loads(request_bytes(req).decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            # Some models (e.g. OpenAI reasoning models) reject `temperature`.
            # React to what the API actually says rather than guessing from
            # model names, and retry ONCE without it. Never silent: warn.
            if e.code == 400 and "temperature" in detail.lower() and "temperature" in body:
                warnings.warn(
                    f"OpenAI rejected temperature={body['temperature']} for model "
                    f"{self.model!r}; retrying without it.",
                    RuntimeWarning, stacklevel=2,
                )
                body.pop("temperature")
                retry = urllib.request.Request(
                    req.full_url, data=json.dumps(body).encode("utf-8"),
                    method="POST", headers=dict(req.headers),
                )
                payload = json.loads(request_bytes(retry).decode("utf-8"))
                return payload["choices"][0]["message"]["content"]
            raise RuntimeError(f"OpenAI request failed: {e.code} {e.reason} — {detail}") from e
        return payload["choices"][0]["message"]["content"]

    def chat_text(self, system: str, user: str) -> str:
        return self._chat(system, user, json_mode=False)

    def chat_json(self, system: str, user: str) -> dict:
        return extract_json(self._chat(system, user, json_mode=True))

    def judge(self, rubric: str, content: str) -> tuple[bool, str]:
        return judge_with(self.chat_json, rubric, content)
