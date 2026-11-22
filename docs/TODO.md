# OpsGate v4: what still needs to be done

Everything that can be done offline is finished (62 tests green, ruff clean, compose config valid). The rest needs your keys, network or Docker.

## Needs you to run
1. `pip install -r requirements.txt`, copy `.env.example` to `.env`, set `GROQ_API_KEY`.
2. `python scripts/seed_db.py && python scripts/make_runbooks.py && python scripts/build_index.py` (real MiniLM embeddings; threshold stays 0.45).
3. `streamlit run app.py` and try the three demo prompts from the README.
4. `python bench/eval_ragas.py` with the key: real RAGAS scores and the real token-reduction number. Offline it measured 78.7% (21 tools); with a real LLM this may differ. Only quote what this run prints.
5. `docker compose up --build` (image never built here).
6. Optional: Langfuse keys, Qdrant (`MEM0_VECTOR_STORE=qdrant`).

## Things to verify with real services (code written, never executed against them)
- Groq tool calling and the three-route prompt; memory extraction JSON.
- FlashRank model download and scores near 0.45; if too strict or loose, tune `SIM_THRESHOLD`.
- `uvx duckduckgo-mcp-server` (tool name `web_search_search`) and live Open-Meteo / Frankfurter responses.
- Mem0 `add(..., infer=False, metadata=...)`, `agent_id` scope and search on your installed `mem0ai` version.
- RAGAS API names differ between versions; `ragas_scores()` falls back to the heuristic if they do not match.

## Open decisions
- Spec says 46 tools; project has 21 (+2 with web search). UI shows the real count; add servers to grow it.
- Benchmark runs 20 of the 24 questions (memory category excluded).
- Light navy theme chosen; say if you want dark.
- Running both pipelines doubles LLM calls (Groq free-tier limits).
- LLM router from v3 removed (rules decide chat / static / ops; everything else RAG then MCP).

## Spec acceptance checklist
- [x] Both pipelines per prompt with tokens_saved / savings_pct
- [x] Runbook question -> hybrid retrieval -> [INTERNAL RUNBOOK]
- [x] Out-of-scope live question -> RAG miss -> MCP escalation (tested with offline fixtures)
- [x] Memory shared across threads (tested)
- [x] PRIVILEGED_WRITE pauses on interrupt() until approved or rejected (tested, UI banner verified headless)
- [x] pytest green offline (62)
- [ ] docker compose up --build (needs Docker)
- [ ] >85% token reduction (needs real LLM run)
