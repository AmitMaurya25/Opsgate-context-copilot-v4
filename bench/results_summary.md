# OpsGate vs Naive: evaluation summary

Questions: 20 (categories: chat,runbook,out_of_scope,ops,long) · Scorer: **heuristic (local fallback, NOT RAGAS)** · LLM: offline fake (numbers are NOT meaningful for quality)

| Metric | Naive baseline | OpsGate |
|---|---|---|
| Faithfulness | 0.74 | 0.75 |
| Answer relevancy | 0.10 | 0.32 |
| Context precision | 0.09 | 0.41 |
| Avg input tokens / question | 1,933 | 412 |
| Avg output tokens / question | 106 | 58 |
| Avg tools in prompt | 21.0 | 2.1 |
| Avg latency (ms) | 2,186 | 958 |

**Measured token reduction: 78.7%** (does NOT meet the >85% target on this run).
Faithfulness maintained or improved.
Numbers come from `bench/results.json`; rerun with a real `GROQ_API_KEY` and real embeddings before quoting them.
