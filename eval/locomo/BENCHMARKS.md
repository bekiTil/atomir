# atomir on LOCOMO

A reproducible harness for evaluating atomir on the
[LOCOMO](https://github.com/snap-research/locomo) long-conversation memory
benchmark, run through
[mem0's memory-benchmarks harness](https://github.com/mem0ai/memory-benchmarks)
via the atomir adapter in this folder.

## What it measures

Per-category answer quality (multi-hop, open-domain, single-hop, temporal) at
each retrieval cutoff, graded by LLM-as-judge (the harness default). atomir uses
its hybrid retrieval (dense + BM25 fused with RRF) with sub-question
decomposition; the episodic layer adds timeline-based temporal answering.

## Reproduce

See [README.md](README.md) for the adapter + run recipe, then aggregate with:

```bash
python eval/locomo/aggregate.py <predicted_dir>
```

Formal benchmark results will be published separately once the full run is
complete.
