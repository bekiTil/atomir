"""Tiny stdlib HTTP helper with retry/backoff for transient failures.

Provider calls (LLM, embedder) share this so a transient failure doesn't fail a
whole request. Two classes of transient error are retried:
  - HTTP status 429/5xx (rate limit / server error) — honors a numeric
    `Retry-After` header when present, else exponential backoff;
  - transport errors (connection reset, timeout, DNS blip) — `URLError` /
    `TimeoutError`, which over thousands of calls are effectively guaranteed.
Non-transient HTTP errors (401/403/400) raise immediately.
"""

from __future__ import annotations

import random
import socket
import time
import urllib.error
import urllib.request

_RETRYABLE = {429, 500, 502, 503, 504}


def _sleep(base: float, max_backoff: float) -> None:
    # Exponential backoff capped, plus jitter (0–50%) so many concurrent callers
    # hitting a rate limit don't retry in lockstep (thundering herd).
    delay = min(base, max_backoff)
    time.sleep(delay + random.uniform(0, delay * 0.5))


def request_bytes(req: urllib.request.Request, *, timeout: float = 60.0,
                  attempts: int = 6, max_backoff: float = 30.0) -> bytes:
    """POST/GET the request and return the response body, retrying transient
    errors (429/5xx and transport blips) with jittered exponential backoff,
    honoring a numeric `Retry-After` when present."""
    for i in range(attempts):
        last = i == attempts - 1
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            # HTTPError is a URLError subclass — handle status codes first.
            if e.code in _RETRYABLE and not last:
                ra = e.headers.get("Retry-After")
                if ra and ra.replace(".", "", 1).isdigit():
                    time.sleep(min(float(ra), max_backoff) + random.uniform(0, 0.5))
                else:
                    _sleep(2 ** i, max_backoff)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError):
            # Transport-level transient errors (reset/timeout/DNS). Back off and
            # retry; on the final attempt let it raise.
            if last:
                raise
            _sleep(2 ** i, max_backoff)
    raise RuntimeError("unreachable")  # pragma: no cover
