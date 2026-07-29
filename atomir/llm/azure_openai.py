"""Azure OpenAI LLM — chat completions via an Azure deployment.

Same stdlib-urllib, no-SDK pattern as the other providers (Azure is a REST API:
`{endpoint}/openai/deployments/{deployment}/chat/completions?api-version=...`,
auth via the `api-key` header). Shared `request_bytes` gives 429/5xx-aware
backoff WITH jitter — required against a TPM ceiling.

gpt-5-mini is a REASONING model: it takes `reasoning_effort` (default "minimal"
so it doesn't burn hidden thinking tokens), uses `max_completion_tokens`, and
does not accept a custom `temperature`. Token usage — including
`reasoning_tokens` — is accumulated on the instance and logged per call so
accounting stays verifiable.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request

from atomir._http import request_bytes
from atomir.llm.parsing import extract_json, judge_with

_log = logging.getLogger("atomir.azure")
_USER_AGENT = "atomir/0.8"


class AzureOpenAILLM:
    def __init__(self, api_key: str, endpoint: str, deployment: str,
                 api_version: str = "2025-01-01-preview",
                 reasoning_effort: str | None = "minimal") -> None:
        if not (api_key and endpoint and deployment):
            raise ValueError(
                "AzureOpenAILLM requires api_key, endpoint, and deployment "
                "(AZURE_OPENAI_KEY / _ENDPOINT / _DEPLOYMENT)."
            )
        self.api_key = api_key
        self.deployment = deployment
        self.api_version = api_version
        self.reasoning_effort = reasoning_effort
        base = endpoint.rstrip("/")
        self.url = (f"{base}/openai/deployments/{deployment}/chat/completions"
                    f"?api-version={api_version}")
        # Cumulative real token usage (verifiable accounting).
        self.usage = {"prompt": 0, "completion": 0, "reasoning": 0, "calls": 0}

    @classmethod
    def from_config(cls, config: dict) -> "AzureOpenAILLM":
        g = lambda k, e: config.get(k) or os.environ.get(e, "")  # config wins, env fallback
        return cls(
            api_key=g("api_key", "AZURE_OPENAI_KEY"),
            endpoint=g("base_url", "AZURE_OPENAI_ENDPOINT"),
            deployment=g("model", "AZURE_OPENAI_DEPLOYMENT"),
            api_version=g("api_version", "AZURE_OPENAI_API_VERSION") or "2025-01-01-preview",
            reasoning_effort=config.get("reasoning_effort",
                                        os.environ.get("AZURE_OPENAI_REASONING_EFFORT", "minimal")),
        )

    def _chat(self, system: str, user: str, json_mode: bool) -> str:
        body: dict = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.reasoning_effort:  # reasoning model: control hidden thinking budget
            body["reasoning_effort"] = self.reasoning_effort
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(
            self.url, data=json.dumps(body).encode("utf-8"), method="POST",
            headers={"Content-Type": "application/json", "api-key": self.api_key,
                     "User-Agent": _USER_AGENT},
        )
        try:
            payload = json.loads(request_bytes(req, timeout=120.0).decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"Azure OpenAI request failed: {e.code} {e.reason} — {detail}") from e
        self._account(payload.get("usage"))
        return payload["choices"][0]["message"]["content"] or ""

    def _account(self, usage: dict | None) -> None:
        if not usage:
            return
        reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
        self.usage["prompt"] += usage.get("prompt_tokens", 0)
        self.usage["completion"] += usage.get("completion_tokens", 0)
        self.usage["reasoning"] += reasoning or 0
        self.usage["calls"] += 1
        _log.debug("azure usage: prompt=%s completion=%s reasoning=%s",
                   usage.get("prompt_tokens"), usage.get("completion_tokens"), reasoning)

    def chat_text(self, system: str, user: str) -> str:
        return self._chat(system, user, json_mode=False)

    def chat_json(self, system: str, user: str) -> dict:
        return extract_json(self._chat(system, user, json_mode=True))

    def judge(self, rubric: str, content: str) -> tuple[bool, str]:
        return judge_with(self.chat_json, rubric, content)
