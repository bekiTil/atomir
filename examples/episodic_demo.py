"""Episodic memory demo — the job-change story.

Feeds a few messages that evolve the user's employer and manager over time, then
asks a temporal + current question and prints the timeline and the composed
answer. Shows the differentiator: facts answer *now*, the event log answers
*when*.

Run with a real LLM + embedder and the episodic flag on:

    EPISODIC_ENABLED=true LLM_BACKEND=openai LLM_API_KEY=sk-... MODEL=gpt-4o-mini \\
    EMBED_BACKEND=openai EMBED_API_KEY=sk-... EMBED_DIM=1536 \\
    STORE_BACKEND=json STORE_PATH=./demo_store.json \\
    python -m examples.episodic_demo

(With the default `fake` backends nothing is extracted — episodic memory needs a
real LLM to turn messages into events.)
"""

from __future__ import annotations

from atomir.assembly import build_memory_service
from atomir.config import settings

MESSAGES = [
    "I joined Beta Inc as an engineer in 2021, and my manager there was Dana.",
    "I left Beta and joined Acme Corp in November 2025.",
    "At Acme my manager is now Sam.",
]


def main() -> None:
    if not settings.episodic_enabled:
        print("Set EPISODIC_ENABLED=true (and real LLM/embedder backends) to run this demo.")
        return

    mem = build_memory_service()
    user = "demo-user"
    mem.reset(user)

    print("Ingesting the job-change story...\n")
    for msg in MESSAGES:
        mem.add(user, msg)
        print(f"  + {msg}")

    print("\n--- timeline: employer over time (when things changed) ---")
    for e in mem.timeline(user, branch="works_at"):
        when = e.get("occurred_at") or e.get("recorded_at", "")[:10]
        print(f"  {when}  {e['polarity']:<6} {e['value']}")

    q = "When did the user switch jobs and who is the user's manager now?"
    print(f"\n--- question: {q}")
    out = mem.answer(user, q)
    print("  sub-questions:", out["subquestions"])
    print("  answer:", out["answer"])


if __name__ == "__main__":
    main()
