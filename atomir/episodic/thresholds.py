"""Per-embedder branch/resolution thresholds — the calibration table the config
reads (and `eval/episodic/branch_microeval.py` writes).

Thresholds are EMBEDDER-DEPENDENT — a value that separates same-branch from
different-branch for one embedder can over-merge on another. Ship reasonable
defaults here; run the micro-eval to calibrate for your embedder and it drops a
`threshold_defaults.json` next to this file that overrides these values.
"""

from __future__ import annotations

import json
import os

# name -> {auto, gray_low, floor, margin}
_HARDCODED = {
    "openai": {"auto": 0.72, "gray_low": 0.40, "floor": 0.30, "margin": 0.10},
    "jina": {"auto": 0.80, "gray_low": 0.40, "floor": 0.30, "margin": 0.10},
}
_FALLBACK = {"auto": 0.80, "gray_low": 0.40, "floor": 0.30, "margin": 0.10}
_JSON = os.path.join(os.path.dirname(__file__), "threshold_defaults.json")


def _table() -> dict:
    table = {k: dict(v) for k, v in _HARDCODED.items()}
    if os.path.exists(_JSON):
        try:
            with open(_JSON, "r", encoding="utf-8") as f:
                for name, vals in json.load(f).items():
                    table.setdefault(name, dict(_FALLBACK)).update(vals)
        except Exception:
            pass
    return table


def get_thresholds(embedder: str) -> dict:
    """Calibrated thresholds for an embedder, falling back to a safe default."""
    return _table().get((embedder or "").lower(), dict(_FALLBACK))


def write_defaults(recommendations: dict) -> str:
    """Persist calibrated per-embedder thresholds (the micro-eval calls this)."""
    existing = {}
    if os.path.exists(_JSON):
        try:
            with open(_JSON, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    existing.update(recommendations)
    with open(_JSON, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, sort_keys=True)
    return _JSON
