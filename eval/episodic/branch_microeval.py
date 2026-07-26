"""Per-embedder branch micro-eval + threshold calibration.

~100 labeled pairs in four families:
  - MERGE        : same-branch paraphrases that MUST merge
  - ANTONYM      : opposite directions of the same relation (same branch: joined/left)
  - SHARED_NAME  : different-branch phrases sharing an entity name — must NOT merge
  - RESOLVE      : planner hint -> emergent branch name (must resolve to the right one)

Runnable against ANY configured embedder in one command; recommends
BRANCH_MATCH_AUTO / BRANCH_MATCH_GRAY_LOW / BRANCH_RESOLVE_FLOOR /
BRANCH_RESOLVE_MARGIN per embedder and (with --write) drops them into the
thresholds table the config reads.

Run:  python -m eval.episodic.branch_microeval [--write]
Uses the .env embedder; adds openai if OPENAI_API_KEY is set.
"""

from __future__ import annotations

import os
import sys

from atomir.episodic.registry import cosine, strip_names

# --- labeled data (branch matching uses name-stripped "phrase (subj -> obj)") ---
# (verb_a, verb_b, shared_name, subj, obj)
MERGE = [
    ("joined", "started at", "Acme", "person", "organization"),
    ("joined", "was hired at", "Acme", "person", "organization"),
    ("works at", "is employed by", "Acme", "person", "organization"),
    ("moved to", "relocated to", "Paris", "person", "place"),
    ("lives in", "resides in", "Paris", "person", "place"),
    ("reports to", "is managed by", "Dana", "person", "person"),
    ("manages", "is the manager of", "Dana", "person", "person"),
    ("owns", "bought", "a Tesla", "person", "object"),
    ("drives", "owns a car made by", "Tesla", "person", "object"),
    ("married to", "is the spouse of", "Sam", "person", "person"),
    ("dating", "is in a relationship with", "Sam", "person", "person"),
    ("studied at", "attended", "Oxford", "person", "organization"),
    ("graduated from", "got a degree at", "Oxford", "person", "organization"),
    ("speaks", "is fluent in", "French", "person", "language"),
    ("plays", "performs on", "guitar", "person", "object"),
    ("took up", "started practicing", "yoga", "person", "activity"),
    ("reads", "subscribes to", "the Times", "person", "media"),
    ("adopted", "got", "a dog", "person", "animal"),
    ("diagnosed with", "suffers from", "asthma", "person", "condition"),
    ("member of", "belongs to", "the club", "person", "organization"),
]
ANTONYM = [  # same branch, opposite direction -> MUST still merge
    ("joined", "left", "Acme", "person", "organization"),
    ("joined", "quit", "Acme", "person", "organization"),
    ("hired at", "resigned from", "Acme", "person", "organization"),
    ("moved to", "moved away from", "Paris", "person", "place"),
    ("started", "stopped", "yoga", "person", "activity"),
    ("took up", "gave up", "guitar", "person", "object"),
    ("adopted", "gave away", "a dog", "person", "animal"),
    ("started dating", "broke up with", "Sam", "person", "person"),
]
SHARED_NAME = [  # different branch, shared name -> must NOT merge
    ("reports to", "reports_to", "is friends with", "friend_of", "Dana", "person", "person"),
    ("works at", "works_at", "invested in", "investor_in", "Acme", "person", "organization"),
    ("lives in", "lives_in", "was born in", "born_in", "Springfield", "person", "place"),
    ("married to", "married_to", "co-founded with", "cofounder", "Sam", "person", "person"),
    ("manages", "manages", "had lunch with", "ate_with", "Priya", "person", "person"),
    ("owns a house in", "owns_in", "commutes to", "commutes_to", "Boston", "person", "place"),
    ("studied at", "studied_at", "donated to", "donated_to", "Oxford", "person", "organization"),
    ("plays for", "plays_for", "supports", "supports", "Rovers", "person", "organization"),
    ("borrowed from", "borrowed_from", "is dating", "dating", "Chris", "person", "person"),
    ("teaches at", "teaches_at", "graduated from", "graduated_from", "Lincoln", "person", "organization"),
    ("sold a car to", "sold_to", "is related to", "related_to", "Morgan", "person", "person"),
    ("drives", "drives", "lives in", "lives_in", "Austin", "person", "place"),  # Austin car? place
]
# planner hint -> the emergent branch's name/description/aliases (must resolve to it,
# and NOT to a distractor). (hint, target_rep, [distractor_reps])
RESOLVE = [
    ("works_at", "joined employment relationship joined left quit hired at",
     ["lives in the place a person lives", "owns something the user owns", "speaks a language"]),
    ("lives_in", "moved_to the place a person lives moved to relocated resides",
     ["works at employment joined left", "plays an instrument", "reads publications"]),
    ("manager_is", "reports_to who the user reports to manager boss managed by",
     ["is friends with a friend", "works at employment", "lives in a place"]),
    ("has_pet", "adopted pet ownership adopted got a dog a cat",
     ["owns a car", "likes a food", "works at a company"]),
    ("studied_at", "graduated_from education studied graduated attended enrolled",
     ["works at employment", "member of a club", "reads publications"]),
]

AUTO_GRID = [i / 100 for i in range(50, 91, 2)]


def _emb_text(verb, subj, obj, name=""):
    return f"{strip_names(verb, {name}) if name else verb} ({subj} -> {obj})"


def evaluate(label, emb):
    cache: dict = {}

    def vec(t):
        if t not in cache:
            cache[t] = emb.embed_passage(t)
        return cache[t]

    def qvec(t):
        key = ("Q", t)
        if key not in cache:
            cache[key] = emb.embed_query(t)
        return cache[key]

    merge_sims, antonym_sims, shared_sims = [], [], []
    for a, b, name, s, o in MERGE:
        merge_sims.append(cosine(vec(_emb_text(a, s, o, name)), vec(_emb_text(b, s, o, name))))
    for a, b, name, s, o in ANTONYM:
        antonym_sims.append(cosine(vec(_emb_text(a, s, o, name)), vec(_emb_text(b, s, o, name))))
    for a, ba, b, bb, name, s, o in SHARED_NAME:
        shared_sims.append(cosine(vec(_emb_text(a, s, o, name)), vec(_emb_text(b, s, o, name))))

    same = merge_sims + antonym_sims        # all MUST-merge (paraphrase + antonym)
    # AUTO: the LOWEST value that keeps 0 shared-name false auto-merges (just above
    # the shared-name ceiling) — so confident same-branch pairs auto-assign while
    # the overlap zone reaches the judge. GRAY_LOW a notch under the merge floor.
    auto = next((t for t in sorted(AUTO_GRID) if sum(s >= t for s in shared_sims) == 0), 0.80)
    gray_low = round(max(0.30, min(same) - 0.05), 2)

    # RESOLVE: floor/margin so hint->target beats distractors.
    tops, margins, ok = [], [], 0
    for hint, target, distractors in RESOLVE:
        hv = qvec(hint.replace("_", " "))
        ts = cosine(hv, vec(target))
        ds = max(cosine(hv, vec(d)) for d in distractors)
        tops.append(ts); margins.append(ts - ds)
        ok += ts > ds
    floor = round(max(0.20, min(tops) - 0.03), 2)
    margin = round(max(0.03, min(margins) - 0.02), 2) if margins else 0.10

    print(f"\n===== {label} =====")
    print(f"  MERGE   n={len(merge_sims)} mean={sum(merge_sims)/len(merge_sims):.3f} min={min(merge_sims):.3f}")
    print(f"  ANTONYM n={len(antonym_sims)} mean={sum(antonym_sims)/len(antonym_sims):.3f} (should be high: same branch)")
    print(f"  SHARED  n={len(shared_sims)} mean={sum(shared_sims)/len(shared_sims):.3f} max={max(shared_sims):.3f} (must stay < AUTO)")
    print(f"  RESOLVE {ok}/{len(RESOLVE)} hint->target beats distractors; top min={min(tops):.3f} margin min={min(margins):.3f}")
    rec = {"auto": auto, "gray_low": gray_low, "floor": floor, "margin": margin}
    print(f"  RECOMMENDED: {rec}")
    return rec


def main():
    write = "--write" in sys.argv
    from atomir.config import settings
    from atomir.providers import EmbedderFactory

    recs = {}
    try:
        recs[settings.embed_backend] = evaluate(settings.embed_backend,
                                                EmbedderFactory.create(settings.embedder))
    except Exception as e:
        print(f"[skip {settings.embed_backend}] {e}")
    key = os.environ.get("OPENAI_API_KEY", "")
    if key and "openai" not in recs:
        from atomir.embeddings.openai import OpenAIEmbedder
        recs["openai"] = evaluate("openai", OpenAIEmbedder(api_key=key, embed_dim=1536))

    print("\n===== RECOMMENDED DEFAULTS =====")
    for name, r in recs.items():
        print(f"  {name}: {r}")
    if write and recs:
        from atomir.episodic.thresholds import write_defaults
        path = write_defaults(recs)
        print(f"\nwrote calibrated thresholds -> {path}")
    elif recs:
        print("\n(dry run; pass --write to persist into the thresholds table)")


if __name__ == "__main__":
    main()
