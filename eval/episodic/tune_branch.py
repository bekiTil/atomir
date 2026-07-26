"""Tune BRANCH_MATCH_AUTO from the branch-matching micro-eval. Sweeps the
auto-assign threshold and, for each value, reports false
auto-merges on the shared-name must-NOT-merge pairs and auto-assign coverage on
the same-branch must-merge pairs. Recommends the LOWEST threshold that yields
ZERO false auto-merges (keeps the most legitimate auto-assigns while staying
safe); everything in the gray band routes to the LLM judge.

Run:  python -m eval.episodic.tune_branch
"""

from __future__ import annotations

import os

from eval.episodic.branch_acceptance import (
    SAME_BRANCH_PAIRS, SHARED_NAME_PAIRS, cosine, _text,
)

SWEEP = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]


def _sims(emb):
    cache: dict = {}

    def sim(text):
        if text not in cache:
            cache[text] = emb.embed_passage(text)
        return cache[text]

    diff = []  # shared-name, must NOT merge (name-stripped)
    for a, ba, b, bb, shared, subj, obj in SHARED_NAME_PAIRS:
        diff.append(cosine(sim(_text(a, shared, subj, obj, True)),
                           sim(_text(b, shared, subj, obj, True))))
    same = []  # same-branch, should auto-assign when possible (name-stripped)
    for a, b, shared, subj, obj in SAME_BRANCH_PAIRS:
        same.append(cosine(sim(_text(a, shared, subj, obj, True)),
                           sim(_text(b, shared, subj, obj, True))))
    return diff, same


def tune(name, emb):
    diff, same = _sims(emb)
    print(f"\n== {name} ==")
    print(f"{'AUTO':<7}{'false-merges':<14}{'same auto-assign':<18}judge-rate")
    recommended = None
    for auto in SWEEP:
        false_merges = sum(s >= auto for s in diff)
        same_auto = sum(s >= auto for s in same)
        judge_rate = sum(0.40 <= s < auto for s in diff) / len(diff)
        print(f"{auto:<7}{false_merges}/{len(diff):<12}{same_auto}/{len(same):<16}{judge_rate:.0%}")
        if recommended is None and false_merges == 0:
            recommended = auto
    print(f"recommended BRANCH_MATCH_AUTO = {recommended}")
    return recommended


def main():
    from atomir.config import settings
    from atomir.providers import EmbedderFactory

    try:
        tune(settings.embed_backend, EmbedderFactory.create(settings.embedder))
    except Exception as e:
        print(f"[skip {settings.embed_backend}] {e}")
    key = os.environ.get("OPENAI_API_KEY", "")
    if key:
        from atomir.embeddings.openai import OpenAIEmbedder
        tune("openai", OpenAIEmbedder(api_key=key, embed_dim=1536))


if __name__ == "__main__":
    main()
