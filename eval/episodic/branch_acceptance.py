"""Acceptance check: does stripping entity names before embedding stop
shared-proper-noun false merges?

Section: >=10 DIFFERENT-branch phrase pairs that SHARE an entity name, labeled
must-NOT-merge. For each embedder we compare two conditions using the real
matcher embed format "{phrase} ({subject_type} -> {object_type})":
  - kept:     names left in  ("reports to Dana ...")
  - stripped: names removed  ("reports to ...")  <- what BranchMatcher does
Acceptance: with stripping + three-zone logic, ZERO pairs land in the
AUTO zone (>= BRANCH_MATCH_AUTO). Gray-zone routing to the judge is fine. If a
pair still auto-merges says lower that embedder's AUTO until it passes
and record the resulting judge-call rate.

Run:  python -m eval.episodic.branch_acceptance
Uses the .env embedder (jina) and, if OPENAI_API_KEY is set, openai too.
"""

from __future__ import annotations

import math
import os

from atomir.episodic.registry import strip_names

# (phrase_a, branch_a, phrase_b, branch_b, shared_name, subj_type, obj_type)
SHARED_NAME_PAIRS = [
    ("reports to Dana", "reports_to", "is friends with Dana", "friend_of", "Dana", "person", "person"),
    ("works at Acme", "works_at", "invested in Acme", "investor_in", "Acme", "person", "organization"),
    ("lives in Springfield", "lives_in", "was born in Springfield", "born_in", "Springfield", "person", "city"),
    ("is married to Sam", "married_to", "co-founded a startup with Sam", "cofounder_with", "Sam", "person", "person"),
    ("manages Priya", "manages", "had lunch with Priya", "ate_with", "Priya", "person", "person"),
    ("owns a house in Boston", "owns_property_in", "commutes to Boston", "commutes_to", "Boston", "person", "city"),
    ("studied at Oxford", "studied_at", "donated to Oxford", "donated_to", "Oxford", "person", "organization"),
    ("is allergic to Rex", "allergic_to", "adopted Rex", "adopted", "Rex", "person", "animal"),
    ("plays for Rovers", "plays_for", "supports Rovers", "supports", "Rovers", "person", "organization"),
    ("borrowed money from Chris", "borrowed_from", "is dating Chris", "dating", "Chris", "person", "person"),
    ("teaches at Lincoln High", "teaches_at", "graduated from Lincoln High", "graduated_from", "Lincoln High", "person", "organization"),
    ("sold a car to Morgan", "sold_to", "is related to Morgan", "related_to", "Morgan", "person", "person"),
]

# Same-branch paraphrases (must MERGE) — to confirm stripping doesn't tank recall.
SAME_BRANCH_PAIRS = [
    ("joined Acme", "started working at Acme", "Acme", "person", "organization"),
    ("reports to Dana", "is managed by Dana", "Dana", "person", "person"),
    ("lives in Paris", "resides in Paris", "Paris", "person", "city"),
    ("owns a Tesla", "bought a Tesla", "Tesla", "person", "vehicle"),
]

AUTO = {"jina": 0.80, "openai": 0.72}  # openai raised per (see config.py)
GRAY_LOW = 0.40


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def _text(phrase, name, subj, obj, strip):
    p = strip_names(phrase, {name}) if strip else phrase
    return f"{p} ({subj} -> {obj})"


def _sim(emb, cache, text):
    if text not in cache:
        cache[text] = emb.embed_passage(text)
    return cache[text]


def _zone(sim, auto):
    return "AUTO" if sim >= auto else ("gray" if sim >= GRAY_LOW else "new")


def evaluate(name, emb):
    auto = AUTO.get(name, 0.80)
    cache: dict = {}
    print(f"\n===== embedder = {name}  (AUTO={auto}, GRAY_LOW={GRAY_LOW}) =====")

    # Shared-name different-branch pairs: must NOT auto-merge.
    print("\nSHARED-NAME must-NOT-merge pairs:  kept -> stripped")
    kept_auto = stripped_auto = stripped_gray = 0
    stripped_sims = []
    for a, ba, b, bb, shared, subj, obj in SHARED_NAME_PAIRS:
        ka = _sim(emb, cache, _text(a, shared, subj, obj, False))
        kb = _sim(emb, cache, _text(b, shared, subj, obj, False))
        sa = _sim(emb, cache, _text(a, shared, subj, obj, True))
        sb = _sim(emb, cache, _text(b, shared, subj, obj, True))
        ksim, ssim = cosine(ka, kb), cosine(sa, sb)
        stripped_sims.append(ssim)
        kept_auto += ksim >= auto
        stripped_auto += ssim >= auto
        stripped_gray += auto > ssim >= GRAY_LOW
        flag = "  <-- FALSE MERGE" if ssim >= auto else ""
        print(f"  {ba:>16} / {bb:<16} {ksim:.3f} -> {ssim:.3f} [{_zone(ssim, auto)}]{flag}")

    n = len(SHARED_NAME_PAIRS)
    print(f"\n  kept:     {kept_auto}/{n} auto-merge (false)")
    print(f"  stripped: {stripped_auto}/{n} auto-merge (false), "
          f"{stripped_gray}/{n} -> gray zone (judge), "
          f"{n - stripped_auto - stripped_gray}/{n} -> new")
    passed = stripped_auto == 0
    print(f"  ACCEPTANCE (0 auto false merges after stripping): "
          f"{'PASS' if passed else 'FAIL'}")
    if not passed:
        need = max(stripped_sims) + 1e-6
        judge_rate = sum(1 for s in stripped_sims if s >= GRAY_LOW) / n
        print(f"  -> lower AUTO to > {need:.3f} for 0 false merges; "
              f"judge-call rate then = {judge_rate:.0%}")

    # Same-branch recall: after stripping they should stay >= GRAY_LOW (auto or judge-recoverable).
    print("\nSAME-BRANCH must-merge pairs (recall after stripping):")
    recall_ok = 0
    for a, b, shared, subj, obj in SAME_BRANCH_PAIRS:
        sa = _sim(emb, cache, _text(a, shared, subj, obj, True))
        sb = _sim(emb, cache, _text(b, shared, subj, obj, True))
        ssim = cosine(sa, sb)
        ok = ssim >= GRAY_LOW
        recall_ok += ok
        print(f"  {a!r} ~ {b!r}: {ssim:.3f} [{_zone(ssim, auto)}]")
    print(f"  {recall_ok}/{len(SAME_BRANCH_PAIRS)} still auto-assign or judge-recoverable")
    return passed


def main():
    from atomir.config import settings
    from atomir.providers import EmbedderFactory

    results = {}
    try:
        results["jina"] = evaluate(settings.embed_backend,
                                   EmbedderFactory.create(settings.embedder))
    except Exception as e:
        print(f"[skip {settings.embed_backend}] {e}")

    key = os.environ.get("OPENAI_API_KEY", "")
    if key:
        from atomir.embeddings.openai import OpenAIEmbedder
        results["openai"] = evaluate("openai", OpenAIEmbedder(api_key=key, embed_dim=1536))

    print("\n===== SUMMARY =====")
    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL (see above)'}")


if __name__ == "__main__":
    main()
