"""Real-provider run of the temporal eval: episodic ON vs OFF, both on real
OpenAI (gpt-4o-mini + text-embedding-3-small). Ingests the same messages through
REAL extraction (no scripting) and measures actual recall / items / tokens per
question. A sanity check that the live pipeline works before the full LoCoMo run.

Run:  python -m eval.episodic.temporal_eval_openai
Needs OPENAI_API_KEY (read from .env). Makes real API calls.
"""

from __future__ import annotations

import os
import tempfile
import time

from dotenv import load_dotenv

from atomir.episodic.engine import EpisodicMemory
from atomir.episodic.json_store import JsonEpisodicStore
from atomir.llm.openai import OpenAILLM
from atomir.embeddings.openai import OpenAIEmbedder
from atomir.memory import MemoryService
from atomir.stores.json_store import JsonMemoryStore
from eval.episodic.temporal_eval import SESSIONS, QUESTIONS


def _tokens(texts):
    return sum(len(t) // 4 for t in texts)


def _recall(gold, texts):
    blob = " ".join(texts).casefold()
    return sum(1 for g in gold if g.casefold() in blob) / len(gold)


def _texts(results):
    return [r.get("text", "") for r in results]


def build_services(tmp, llm, emb):
    pack = os.environ.get("ONTOLOGY_PACK", "")  # empty = feedback-only (general-purpose)
    facts_on = JsonMemoryStore(path=f"{tmp}/facts_on.json")
    ep = JsonEpisodicStore(path=f"{tmp}/ep.json")
    engine = EpisodicMemory(facts_on, ep, llm, emb, branch_auto=0.72, branch_gray_low=0.40,
                            ontology_pack=pack)
    on = MemoryService(facts_on, llm, emb, episodic=engine)

    facts_off = JsonMemoryStore(path=f"{tmp}/facts_off.json")
    off = MemoryService(facts_off, llm, emb)  # classic atomir (episodic off)
    return on, off, engine, ep


def main():
    load_dotenv()  # finds .env in the project root
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        print("No OPENAI_API_KEY."); return
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    print(f"model = {model}")
    llm = OpenAILLM(api_key=key, model=model)
    emb = OpenAIEmbedder(api_key=key, embed_dim=1536)

    tmp = tempfile.mkdtemp()
    on, off, engine, ep = build_services(tmp, llm, emb)

    print("ingesting", len(SESSIONS), "messages through REAL extraction (x2 services)...")
    t0 = time.perf_counter()
    for text, _ in SESSIONS:
        on.add("u", text)
        off.add("u", text)
    print(f"ingest done in {time.perf_counter()-t0:.1f}s\n")

    # B1/B2 smoke check: employment must be ONE branch with start+end+start.
    self_e = ep.entity_by_alias("u", "the user")
    print("USER branches with events:")
    for b in ep.branches("u", self_e.entity_id):
        evs = ep.events("u", entity_id=self_e.entity_id, branch=b.branch)
        if evs:
            print(f"  {b.branch:<18} {[ (e.polarity, e.value) for e in evs ]}")
    print()

    rows = []
    for q in QUESTIONS:
        res_on = on.search("u", q["q"])
        r_on = res_on["results"]
        r_off = off.search("u", q["q"])["results"]
        on_t, off_t = _texts(r_on), _texts(r_off)
        if q["type"] == "temporal":
            print(f"  [{q['q'][:40]}] branch_resolved={res_on.get('branch_resolved')} "
                  f"fallback_used={res_on.get('fallback_used')}")
        rows.append({
            "q": q["q"], "type": q["type"], "gold": q["gold"],
            "on_items": len(on_t), "on_tokens": _tokens(on_t), "on_recall": _recall(q["gold"], on_t),
            "off_items": len(off_t), "off_tokens": _tokens(off_t), "off_recall": _recall(q["gold"], off_t),
            "on_texts": on_t, "off_texts": off_t,
        })

    print(f"{'type':<10}{'question':<52}{'ON i/tok/rec':<20}{'OFF i/tok/rec'}")
    print("-" * 110)
    for r in rows:
        on_cell = f"{r['on_items']}/{r['on_tokens']}/{r['on_recall']:.2f}"
        off_cell = f"{r['off_items']}/{r['off_tokens']}/{r['off_recall']:.2f}"
        print(f"{r['type']:<10}{r['q'][:50]:<52}{on_cell:<20}{off_cell}")
    print("\n--- retrieved detail ---")
    for r in rows:
        print(f"\nQ: {r['q']}  (gold={r['gold']})")
        print("  ON :", r["on_texts"])
        print("  OFF:", r["off_texts"])

    temp = [r for r in rows if r["type"] == "temporal"]
    def rpt(rs, ts):
        r = sum(x[rs] for x in temp)/len(temp); t = sum(x[ts] for x in temp)/len(temp)
        return r, t
    onr, ont = rpt("on_recall", "on_tokens")
    offr, offt = rpt("off_recall", "off_tokens")
    print("\n=== temporal summary ===")
    print(f"  episodic on : recall {onr:.2f}, {ont:.0f} tokens")
    print(f"  episodic off: recall {offr:.2f}, {offt:.0f} tokens")


if __name__ == "__main__":
    main()
