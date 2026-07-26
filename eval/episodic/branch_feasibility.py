"""Feasibility check for branch matching.

Question this answers, on the REAL embedder configured in .env (not the
hash-based fake): can embedding similarity separate *same-branch* verb phrasings
from *different-branch* near-neighbors well enough that the
"embeddings nominate, LLM decides" branch matcher is viable?

We label short predicate phrases with a canonical branch, then:
  1. embed each phrase once (cached),
  2. score every intra-branch pair (SHOULD be high -> "nominate as same branch")
     and every cross-branch NEAR-NEIGHBOR pair (SHOULD be lower),
  3. sweep a threshold and report precision/recall/F1 + the gray-zone overlap
     band (where the LLM judge must arbitrate),
  4. separately report ANTONYM pairs (same branch, opposite polarity, e.g.
     joined/left): the spec claims these embed close, which HELPS branch merging
     while polarity carries direction.

Run:  python -m eval.episodic.branch_feasibility
Uses whatever EMBED_BACKEND/.env is active. No LLM calls, no writes.
"""

from __future__ import annotations

import itertools
import math

from atomir.config import settings
from atomir.providers import EmbedderFactory

# branch -> phrasings. "start"/"end" mark polarity within the SAME branch;
# antonyms (end vs start) are same-branch per spec (polarity is a separate axis).
BRANCHES: dict[str, list[tuple[str, str]]] = {
    "works_at": [
        ("start", "joined Acme Corp"),
        ("start", "started working at Acme Corp"),
        ("start", "was hired at Acme Corp"),
        ("start", "began a job at Acme Corp"),
        ("end", "left Acme Corp"),
        ("end", "quit Acme Corp"),
        ("end", "resigned from Acme Corp"),
    ],
    "lives_in": [
        ("start", "moved to Paris"),
        ("start", "now lives in Paris"),
        ("start", "relocated to Paris"),
        ("start", "resides in Paris"),
        ("end", "moved away from Paris"),
    ],
    "reports_to": [
        ("start", "reports to Dana"),
        ("start", "is managed by Dana"),
        ("start", "works under Dana"),
        ("start", "Dana is the manager"),
        ("end", "no longer reports to Dana"),
    ],
    "likes_activity": [
        ("start", "likes running"),
        ("start", "enjoys running"),
        ("start", "is into running"),
        ("end", "dislikes running"),
    ],
    "owns_vehicle": [
        ("start", "owns a Tesla"),
        ("start", "bought a Tesla"),
        ("start", "purchased a Tesla"),
        ("end", "sold the Tesla"),
    ],
}

# Cross-branch pairs that are deliberately CONFUSABLE (share a token or verb, or
# concern the same entity) — the matcher MUST keep these on different branches.
NEAR_NEIGHBORS: list[tuple[str, str]] = [
    ("works at Acme Corp", "lives in Acme City"),        # shared token "Acme"
    ("joined Acme Corp", "joined a running club"),        # same verb, diff object
    ("reports to Dana", "is friends with Dana"),          # same person, diff relation
    ("moved to Paris", "moved on from that job"),         # same verb, diff domain
    ("owns a Tesla", "drives to work"),                   # both vehicle-ish
    ("likes running", "runs a company"),                  # token "run", diff sense
    ("bought a Tesla", "bought a coffee this morning"),   # durable vs transient
    ("is managed by Dana", "manages a small team"),       # manage both ways
]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def main() -> None:
    emb = EmbedderFactory.create(settings.embedder)
    print(f"embedder backend = {settings.embed_backend}  dim = {settings.embed_dim}\n")

    # Collect + embed every unique phrase once.
    phrases: set[str] = set()
    for items in BRANCHES.values():
        phrases.update(p for _, p in items)
    for a, b in NEAR_NEIGHBORS:
        phrases.update((a, b))
    vec: dict[str, list[float]] = {p: emb.embed_passage(p) for p in sorted(phrases)}

    # Positive pairs: same branch. Track antonym (start/end) pairs separately.
    same_branch: list[tuple[float, str]] = []
    antonym: list[tuple[float, str]] = []
    for branch, items in BRANCHES.items():
        for (pa, ta), (pb, tb) in itertools.combinations(items, 2):
            s = cosine(vec[ta], vec[tb])
            label = f"{branch}: {ta!r} ~ {tb!r}"
            same_branch.append((s, label))
            if pa != pb:  # opposite polarity within same branch
                antonym.append((s, label))

    # Negative pairs: cross-branch confusables.
    diff_branch = [(cosine(vec[a], vec[b]), f"{a!r} ~ {b!r}") for a, b in NEAR_NEIGHBORS]

    def stats(pairs: list[tuple[float, str]]) -> str:
        xs = sorted(s for s, _ in pairs)
        n = len(xs)
        mean = sum(xs) / n
        med = xs[n // 2]
        return f"n={n}  min={xs[0]:.3f}  med={med:.3f}  mean={mean:.3f}  max={xs[-1]:.3f}"

    print("SAME-branch pairs (should be HIGH):   ", stats(same_branch))
    print("DIFF-branch near-neighbors (LOW):     ", stats(diff_branch))
    print("  of which ANTONYM (same branch):     ", stats(antonym))

    # Threshold sweep: classify "same branch" if sim >= t. Report P/R/F1.
    pos = [s for s, _ in same_branch]
    neg = [s for s, _ in diff_branch]
    print("\n  t     precision  recall   F1     (same-branch=positive)")
    best = (0.0, 0.0)
    for t in [i / 100 for i in range(30, 96, 5)]:
        tp = sum(1 for s in pos if s >= t)
        fp = sum(1 for s in neg if s >= t)
        fn = sum(1 for s in pos if s < t)
        prec = tp / (tp + fp) if (tp + fp) else 1.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if f1 > best[1]:
            best = (t, f1)
        print(f"  {t:.2f}    {prec:.2f}       {rec:.2f}     {f1:.2f}")
    print(f"\n  best F1 = {best[1]:.2f} at t = {best[0]:.2f}")

    # Gray zone = overlap band where positives and negatives coexist.
    lo = min(pos)          # lowest same-branch score
    hi = max(neg)          # highest different-branch score
    print(f"\n  overlap / gray zone: diff-branch max = {hi:.3f}, "
          f"same-branch min = {lo:.3f}")
    if hi < lo:
        print("  -> CLEAN SEPARATION: a single threshold splits them "
              "(LLM judge rarely needed).")
    else:
        n_pos_in = sum(1 for s in pos if lo <= s <= hi)
        n_neg_in = sum(1 for s in neg if lo <= s <= hi)
        print(f"  -> OVERLAP: {n_pos_in} same-branch and {n_neg_in} diff-branch "
              f"pairs fall in [{lo:.3f}, {hi:.3f}].")
        print("     Pure threshold cannot separate these -> LLM judge in the gray "
              "zone is JUSTIFIED (validates the spec's design).")

    print("\n  worst 5 same-branch (hardest to merge):")
    for s, lbl in sorted(same_branch)[:5]:
        print(f"    {s:.3f}  {lbl}")
    print("  top 5 diff-branch (most likely FALSE merges):")
    for s, lbl in sorted(diff_branch, reverse=True)[:5]:
        print(f"    {s:.3f}  {lbl}")


if __name__ == "__main__":
    main()
