"""Distractor-scale temporal eval.

The 11-message eval can't test the efficiency claim: baseline top-k spans half a
12-fact store, so it trivially hits gold. Here one user gets ~150 messages — a
handful of GOLD messages encoding evolving states (employment x3, city x2,
manager x2) interleaved chronologically with ~140 deterministic distractors — so
branch walks (O(one chain)) can actually be cheaper than top-k over a large
store. Metrics: recall, precision@k (punishes the firehose), items, tokens
(tiktoken), and temporal fallback rate.

Offline smoke:  python -m eval.episodic.distractor_eval --offline
Real matrix:    OPENAI_MODELS=gpt-4o-mini,gpt-4o python -m eval.episodic.distractor_eval
                (prints a budget estimate; set RUN=1 to execute real runs)
"""

from __future__ import annotations

import os
import random
import sys
import tempfile
import time

from atomir.embeddings.fake import FakeEmbedder
from atomir.llm.fake import FakeLLM
from atomir.episodic.engine import EpisodicMemory
from atomir.episodic.json_store import JsonEpisodicStore
from atomir.memory import MemoryService
from atomir.stores.json_store import JsonMemoryStore

# --- corpus ---------------------------------------------------------------

GOLD = [
    ("2021-02-01", "I joined Beta Inc as a software engineer."),
    ("2021-02-10", "My manager at Beta is Dana."),
    ("2021-03-01", "I moved to Boston for the new job."),
    ("2022-04-01", "I took up landscape painting as a hobby."),
    ("2022-09-01", "My favorite food is sushi."),
    ("2023-05-01", "I left Beta and joined Acme Corp."),
    ("2023-05-10", "At Acme my manager is now Sam."),
    ("2024-06-01", "I relocated from Boston to Denver."),
    ("2025-01-15", "I left Acme and joined Cognitech."),
]

QUESTIONS = [
    # temporal (10)
    ("temporal", "Which companies has the user worked for over time?", ["Beta", "Acme", "Cognitech"]),
    ("temporal", "Who was the user's first employer?", ["Beta"]),
    ("temporal", "Where did the user work right before Acme?", ["Beta"]),
    ("temporal", "Who did the user work for in 2024?", ["Acme"]),
    ("temporal", "Which cities has the user lived in?", ["Boston", "Denver"]),
    ("temporal", "Where did the user live before Denver?", ["Boston"]),
    ("temporal", "Who have been the user's managers over time?", ["Dana", "Sam"]),
    ("temporal", "Who was the user's manager before Sam?", ["Dana"]),
    ("temporal", "What was the user's first city?", ["Boston"]),
    ("temporal", "Which employer came after Beta?", ["Acme"]),
    # current (5)
    ("current", "What company does the user work for now?", ["Cognitech"]),
    ("current", "What city does the user currently live in?", ["Denver"]),
    ("current", "Who is the user's manager now?", ["Sam"]),
    ("current", "Where does the user live now?", ["Denver"]),
    ("current", "Who is the user's current employer?", ["Cognitech"]),
    # semantic (5)
    ("semantic", "What hobby did the user take up?", ["painting"]),
    ("semantic", "What is the user's favorite food?", ["sushi"]),
    ("semantic", "Does the user paint?", ["painting"]),
    ("semantic", "What does the user like to eat?", ["sushi"]),
    ("semantic", "What creative hobby does the user have?", ["painting"]),
]

_FOODS = ["tacos", "ramen", "pho", "curry", "pizza", "dumplings", "falafel", "bibimbap"]
_PLACES = ["Austin", "Miami", "Seattle", "Chicago", "Lisbon", "Kyoto", "Berlin"]
_ACTS = ["a museum", "a concert", "the farmers market", "a book club", "a 10k race"]
_THINGS = ["a new phone", "a bike", "headphones", "a coffee grinder", "a plant"]
_NAMES = ["Priya", "Marcus", "Lena", "Omar", "Sofia", "Kenji"]


def generate_corpus(n=150, seed=7):
    rng = random.Random(seed)
    msgs = list(GOLD)
    templates = [
        lambda: f"I tried {rng.choice(_FOODS)} for dinner.",
        lambda: f"I visited {rng.choice(_PLACES)} for the weekend.",
        lambda: f"I went to {rng.choice(_ACTS)} on Saturday.",
        lambda: f"I bought {rng.choice(_THINGS)}.",
        lambda: f"I caught up with my friend {rng.choice(_NAMES)}.",
        lambda: f"The weather was nice so I went for a walk.",
        lambda: f"I watched a documentary about {rng.choice(_PLACES)}.",
    ]
    for _ in range(max(0, n - len(GOLD))):
        y = rng.randint(2021, 2025); m = rng.randint(1, 12); d = rng.randint(1, 28)
        msgs.append((f"{y}-{m:02d}-{d:02d}", rng.choice(templates)()))
    msgs.sort(key=lambda t: t[0])              # chronological interleave
    return [m for _, m in msgs]


# --- metrics --------------------------------------------------------------

def _token_counter():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return lambda t: len(enc.encode(t))
    except Exception:
        return lambda t: max(1, len(t) // 4)


def _relevant(text, gold):
    blob = text.casefold()
    return any(g.casefold() in blob for g in gold)


# --- run ------------------------------------------------------------------

# mode -> (episodic?, ontology_pack)
MODES = {"OFF": (False, ""), "episodic": (True, ""), "episodic-pack": (True, "personal")}


def build(tmp, llm, emb, mode):
    facts = JsonMemoryStore(path=f"{tmp}/{mode}_facts.json")
    episodic, pack = MODES[mode]
    if not episodic:
        return MemoryService(facts, llm, emb)  # classic atomir
    ep = JsonEpisodicStore(path=f"{tmp}/{mode}_ep.json")
    engine = EpisodicMemory(facts, ep, llm, emb, branch_auto=0.72,
                            branch_gray_low=0.40, ontology_pack=pack)
    return MemoryService(facts, llm, emb, episodic=engine)


def run_config(llm, emb, mode, messages, k=6):
    tmp = tempfile.mkdtemp()
    svc = build(tmp, llm, emb, mode)
    t0 = time.perf_counter()
    for m in messages:
        svc.add("u", m)
    ingest_s = time.perf_counter() - t0

    tok = _token_counter()
    by_type: dict = {}
    mechs: dict = {}   # resolution mechanism counts on temporal questions
    for typ, q, gold in QUESTIONS:
        res = svc.search("u", q, k=k)
        texts = [r.get("text", "") for r in res["results"]]
        recall = sum(1 for g in gold if any(g.casefold() in t.casefold() for t in texts)) / len(gold)
        prec = (sum(_relevant(t, gold) for t in texts) / len(texts)) if texts else 0.0
        toks = sum(tok(t) for t in texts)
        if typ == "temporal":
            m = res.get("resolve_mechanism") or "n/a"
            mechs[m] = mechs.get(m, 0) + 1
        b = by_type.setdefault(typ, {"recall": [], "prec": [], "items": [], "toks": []})
        b["recall"].append(recall); b["prec"].append(prec)
        b["items"].append(len(texts)); b["toks"].append(toks)

    def avg(xs): return sum(xs) / len(xs) if xs else 0.0
    n_temporal = sum(1 for t, _, _ in QUESTIONS if t == "temporal")
    summary = {t: {kk: avg(vv) for kk, vv in d.items()} for t, d in by_type.items()}
    summary["_ingest_s"] = ingest_s
    summary["_mechanisms"] = mechs   # e.g. {"exact": 8, "relative-best": 1, "fallback": 1}
    return summary


def _mech_str(mechs: dict) -> str:
    if not mechs:
        return "n/a"
    return ", ".join(f"{k}:{v}" for k, v in sorted(mechs.items(), key=lambda t: -t[1]))


def render(results: dict, n_msgs: int) -> str:
    out = [f"# Distractor-scale temporal eval ({n_msgs} messages)\n",
           "| config | qtype | recall | precision@k | items | tokens | temporal resolution |",
           "|---|---|---|---|---|---|---|"]
    for name, s in results.items():
        for typ in ("temporal", "current", "semantic"):
            if typ in s:
                d = s[typ]
                mech = _mech_str(s.get("_mechanisms", {})) if typ == "temporal" else ""
                out.append(f"| {name} | {typ} | {d['recall']:.2f} | {d['prec']:.2f} "
                           f"| {d['items']:.1f} | {d['toks']:.0f} | {mech} |")
        out.append(f"| {name} | _meta_ | ingest {s['_ingest_s']:.0f}s | | | | |")
    return "\n".join(out)


def _providers(model):
    from dotenv import load_dotenv
    load_dotenv()
    key = os.environ["OPENAI_API_KEY"]
    from atomir.llm.openai import OpenAILLM
    from atomir.embeddings.openai import OpenAIEmbedder
    return OpenAILLM(api_key=key, model=model), OpenAIEmbedder(api_key=key, embed_dim=1536)


def main(argv=None):
    argv = argv or sys.argv[1:]
    n = int(os.environ.get("DISTRACTOR_N", "150"))
    messages = generate_corpus(n)

    configs = ["OFF", "episodic", "episodic-pack"]
    if "--offline" in argv:  # plumbing smoke, fake providers
        res = {f"{c}(fake)": run_config(FakeLLM(), FakeEmbedder(), c, messages) for c in configs}
        print(render(res, len(messages)))
        return res

    models = os.environ.get("OPENAI_MODELS", "gpt-4o-mini").split(",")
    n_runs = len(models) * len(configs)
    est_tokens = (len(messages) * 400 + len(QUESTIONS) * 300) * n_runs
    print(f"BUDGET ESTIMATE: ~{est_tokens:,} tokens across {n_runs} runs "
          f"({len(models)} model(s) x {len(configs)} configs, {len(messages)} msgs).")
    if os.environ.get("RUN") != "1":
        print("Dry run. Set RUN=1 to execute the real matrix."); return
    results = {}
    for model in models:
        llm, emb = _providers(model.strip())
        for mode in configs:
            print(f"running {model} {mode} ...", flush=True)
            results[f"{model}/{mode}"] = run_config(llm, emb, mode, messages)
    report = render(results, len(messages))
    print("\n" + report)
    with open(os.path.join(os.path.dirname(__file__), "RESULTS.md"), "w") as f:
        f.write(report + "\n")
    print("\nwrote eval/episodic/RESULTS.md")
    return results


if __name__ == "__main__":
    main()
