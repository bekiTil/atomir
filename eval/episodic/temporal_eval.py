"""Temporal retrieval efficiency eval.

The claim: for temporal / historical questions, an episodic chain walk returns a
FEW precise events, whereas flat semantic/hybrid retrieval must pull top-k facts
that (a) cost far more tokens and (b) can MISS the answer entirely — because a
superseded state (e.g. a former employer) is no longer a live fact, it only
exists in the event log. So episodic wins on the headline **recall per retrieved
token**.

This harness is self-contained and deterministic offline (scripted extraction +
fake embedder), so it runs in CI with no keys. The SAME structure accepts real
providers (build the services from a real LLM/embedder) and real datasets
(LongMemEval / LoCoMo temporal via the eval/locomo adapter) for an accuracy
comparison against mem0 — that part needs your own keys and is not run here.

Run:  python -m eval.episodic.temporal_eval
"""

from __future__ import annotations

import tempfile
import time

from atomir.embeddings.fake import FakeEmbedder
from atomir.atomic_read import atomic_search
from atomir.episodic.engine import EpisodicMemory
from atomir.episodic.json_store import JsonEpisodicStore
from atomir.stores.json_store import JsonMemoryStore

BASELINE_K = 10  # how many facts a flat retriever pulls into context


def _ev(verb, value, polarity="start", occurred_at=None, obj="thing"):
    return {"verb_phrase": verb, "value": value, "subject": "the user",
            "subject_type": "person", "object_type": obj, "polarity": polarity,
            "modality": "happened", "occurred_at": occurred_at,
            "raw_text": f"{verb} {value}"}


def _branch(state_template, description):
    return {"state_template": state_template, "description": description}


# Each verb phrase maps to a branch definition (used to script the namer).
VERB_BRANCH = {
    "works at": _branch("works at", "employment"),
    "has a pet": _branch("has a pet", "pet ownership"),
    "enjoys": _branch("enjoys", "hobby"),
    "lives in": _branch("lives in", "residence"),
    "drives": _branch("drives", "vehicle"),
    "plays": _branch("plays", "instrument"),
    "likes eating": _branch("likes eating", "food preference"),
    "speaks": _branch("speaks", "language"),
    "reads": _branch("reads", "publication"),
    "uses": _branch("uses", "equipment"),
}

SESSIONS = [
    ("I joined Beta Inc back in 2023.",
     [_ev("works at", "Beta Inc", "start", "2023-01-01", "organization")]),
    ("I left Beta and joined Acme Corp in November 2025.",
     [_ev("works at", "Beta Inc", "end", "2025-11-01", "organization"),
      _ev("works at", "Acme Corp", "start", "2025-11-01", "organization")]),
    # noise: each grows the flat fact pool with an unrelated current-state fact
    ("I have a dog named Rex.", [_ev("has a pet", "a dog named Rex", obj="animal")]),
    ("I took up rock climbing.", [_ev("enjoys", "rock climbing", obj="hobby")]),
    ("I live in Portland.", [_ev("lives in", "Portland", obj="city")]),
    ("I drive a Tesla.", [_ev("drives", "a Tesla", obj="vehicle")]),
    ("I play the guitar.", [_ev("plays", "the guitar", obj="instrument")]),
    ("I like eating sushi.", [_ev("likes eating", "sushi", obj="food")]),
    ("I speak French.", [_ev("speaks", "French", obj="language")]),
    ("I read the New Yorker.", [_ev("reads", "the New Yorker", obj="publication")]),
    ("I use a standing desk.", [_ev("uses", "a standing desk", obj="equipment")]),
]

QUESTIONS = [
    {"q": "Which companies has the user worked for over time?", "type": "temporal",
     "entity": "the user", "branch": "works_at", "until": None, "gold": ["Beta", "Acme"]},
    {"q": "Who did the user work for in 2024?", "type": "temporal",
     "entity": "the user", "branch": "works_at", "until": "2024-12-31", "gold": ["Beta"]},
    {"q": "Where does the user live now?", "type": "current",
     "entity": None, "branch": None, "until": None, "gold": ["Portland"]},
]


class _ScriptLLM:
    """Deterministic LLM: pops scripted extractions and names branches by verb."""

    def __init__(self, extracts):
        self._extracts = list(extracts)

    def chat_json(self, system, user):
        if "extract EVENTS" in system:
            return self._extracts.pop(0)
        if "name a NEW predicate branch" in system:
            verb = user.split("PREDICATE PHRASE:", 1)[1].split("\n", 1)[0].strip()
            b = VERB_BRANCH.get(verb, _branch(verb, verb))
            return {"branch": verb.replace(" ", "_"), **b}
        if "branch matcher" in system:
            return {"branch": "NEW"}
        return {}

    def chat_text(self, system, user):
        return ""


def _tokens(texts):
    return sum(len(t) // 4 for t in texts)


def _recall(gold, texts):
    blob = " ".join(texts).casefold()
    return sum(1 for g in gold if g.casefold() in blob) / len(gold)


def run_eval():
    tmp = tempfile.mkdtemp()
    facts = JsonMemoryStore(path=f"{tmp}/facts.json")
    ep = JsonEpisodicStore(path=f"{tmp}/ep.json")
    llm = _ScriptLLM([{"events": evs} for _, evs in SESSIONS])
    emb = FakeEmbedder()
    mem = EpisodicMemory(facts, ep, llm, emb, branch_auto=0.8, branch_gray_low=0.4)
    for text, _ in SESSIONS:
        mem.add("u", text)

    rows = []
    for q in QUESTIONS:
        # Episodic route: temporal -> chain walk; current -> current facts.
        t0 = time.perf_counter()
        if q["type"] == "temporal":
            ev = mem.timeline("u", entity=q["entity"], branch=q["branch"], until=q["until"])
            epi_texts = [e["text"] for e in ev]
        else:
            hits = atomic_search(facts, llm, emb, "u", q["q"], decompose=False, hybrid=True, k=6)
            epi_texts = [r["text"] for r in hits["results"]]
        epi_ms = (time.perf_counter() - t0) * 1000

        # Baseline: flat hybrid retrieval over the same facts (episodic-off).
        t0 = time.perf_counter()
        base = atomic_search(facts, llm, emb, "u", q["q"], decompose=False,
                             hybrid=True, k=BASELINE_K)
        base_texts = [r["text"] for r in base["results"]]
        base_ms = (time.perf_counter() - t0) * 1000

        rows.append({
            "q": q["q"], "type": q["type"], "gold": q["gold"],
            "epi_items": len(epi_texts), "epi_tokens": _tokens(epi_texts),
            "epi_recall": _recall(q["gold"], epi_texts), "epi_ms": epi_ms,
            "base_items": len(base_texts), "base_tokens": _tokens(base_texts),
            "base_recall": _recall(q["gold"], base_texts), "base_ms": base_ms,
        })
    return rows


def _rpt(recall, tokens):
    return recall / (tokens / 1000) if tokens else 0.0  # recall per 1k tokens


def render(rows) -> str:
    out = ["# Temporal retrieval efficiency (episodic vs flat hybrid)\n",
           "| question | type | episodic items/tokens/recall | baseline items/tokens/recall |",
           "|---|---|---|---|"]
    for r in rows:
        out.append(
            f"| {r['q']} | {r['type']} | "
            f"{r['epi_items']} / {r['epi_tokens']} / {r['epi_recall']:.2f} | "
            f"{r['base_items']} / {r['base_tokens']} / {r['base_recall']:.2f} |")
    temporal = [r for r in rows if r["type"] == "temporal"]
    er = sum(r["epi_recall"] for r in temporal) / len(temporal)
    et = sum(r["epi_tokens"] for r in temporal) / len(temporal)
    br = sum(r["base_recall"] for r in temporal) / len(temporal)
    bt = sum(r["base_tokens"] for r in temporal) / len(temporal)
    out += ["", "## Headline (temporal questions)",
            f"- episodic: recall {er:.2f} at {et:.0f} tokens -> **{_rpt(er, et):.2f} recall / 1k tokens**",
            f"- baseline: recall {br:.2f} at {bt:.0f} tokens -> {_rpt(br, bt):.2f} recall / 1k tokens",
            "",
            "Note: offline/self-contained. Plug in real providers + LongMemEval/"
            "LoCoMo temporal (eval/locomo adapter) and mem0 for an accuracy-vs-mem0 run."]
    return "\n".join(out)


def main():
    print(render(run_eval()))


if __name__ == "__main__":
    main()
