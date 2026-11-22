# OpsGate v4: what has been done so far

Status: all spec steps implemented. **62 tests pass offline and `ruff check` is clean.** Real Groq / Mem0 / FlashRank / live network / Docker build are still untested (see TODO.md).
Base: your `opsgate-context-copilot-v3.zip`. Spec followed: `gemini-code-1795111905004.md` (v4.0 master spec).

## Verified by running it (manual smoke tests, offline: fake LLM + hash embedder)
- v3 baseline: all 38 original tests passed before any change.
- 6 MCP servers load from `mcp_servers.json`; 21 tools namespaced as `<server>_<tool>`; `web_search` is skipped cleanly when `uvx` has no network.
- RAG hit path ("How do I fix Error 5042?") -> `[INTERNAL RUNBOOK]`; RAG miss path (weather + currency) -> deferred search -> utility tools -> `[LIVE MCP / WEB SEARCH]`.
- Human approval: a rollback request pauses (`interrupted`), and Approve / Reject both resume correctly.
- Naive and OpsGate run in parallel; `tokens_saved` and `savings_pct` are returned.
- FastAPI endpoints answered correctly through `TestClient` (compare, resume, memory, MCP servers).
- Streamlit chat UI loaded with no errors in a headless run; a chat turn showed both comparison cards, and the approval banner appeared and cleared after Approve.

## Files created (new)
| File | Purpose |
|---|---|
| `mcp_servers.json` | Plug-and-play registry of 6 servers, with `default_risk` and `risk_overrides` |
| `mcp_servers/utility_server.py` | Live weather (Open-Meteo), currency (Frankfurter), world time; `UTILITY_OFFLINE=1` returns canned data for tests |
| `core/guardrails.py` | RBAC (Viewer / OnCall-Engineer / Admin), PII and key redaction, prompt-injection quarantine |
| `core/client.py` | One client for the UI: HTTP to the API, or in-process |
| `api/server.py` | FastAPI: `/v1/chat/compare`, `/v1/chat/resume`, `/v1/memory/{user_id}`, `/v1/mcp/servers` (GET and POST), `/v1/upload`, `/v1/files`, `/v1/session/end`, `/v1/health` |
| `tests/conftest.py` | Offline test environment |
| `tests/test_v4.py` | 24 new tests for the v4 features (written, not yet run) |
| `docs/DONE.md`, `docs/TODO.md` | These notes |

## Files changed (existing)
| File | What changed |
|---|---|
| `core/mcp_host.py` | Reads the JSON registry, strips custom keys, prefixes tool names, tags risk per tool, skips failed servers, 8 s timeout with 1 retry, `register_new_mcp_server()`, hot `reload()` |
| `core/rag_retriever.py` | Hybrid: dense + BM25, RRF fusion, FlashRank rerank (falls back to a dense + lexical-coverage blend when the model is unavailable), threshold gate, returns original chunk text only |
| `core/tool_registry.py` | Risk and server metadata, role-filtered search, multi-intent search |
| `core/graph.py` | Full rewrite: memory recall node, RAG gate, Route A / B / C, agent and tools loop with `interrupt()` approvals, async memory write, SQLite checkpointer, `run_comparison_turn` and `resume_comparison` |
| `core/prompts.py` | Hybrid grounding prompt; memory block with `[SEMANTIC]`, `[EPISODIC]`, `[PROCEDURAL]` bullets |
| `core/llm.py` | New ignorance text ("I cannot verify this from internal runbooks or live MCP tools."); smarter offline fake LLM; hash embedder drops stopwords |
| `core/telemetry.py` | Adds trace, provenance, output tokens, contexts, memory used, role, latency ms |
| `core/tracing.py` | Adds redacted `logs/audit.jsonl`, OpenTelemetry spans (Langfuse export when keys are set) |
| `core/sandbox.py` | Works with prefixed tool names |
| `core/config.py` | New settings (registry path, timeouts, reranker, thresholds, Qdrant, audit paths) |
| `core/threads.py` | Chat log can store the comparison payload |
| `memory/mem0_store.py` | 3 tiers: semantic and episodic by `user_id`, procedural by `agent_id="opsgate-copilot"`; not scoped by thread; Qdrant option |
| `memory/manager.py` | Per-turn recall across tiers, classifier, dedup before write, async write |
| `memory/dedup.py` | Added `is_duplicate` |
| `mcp_servers/incident_server.py` | Added privileged tools `restart_worker` and `merge_pull_request` |
| `ingestion/contextual_indexer.py` | Thread-safe Chroma client (needed for parallel pipelines) |
| `app.py` | New professional chat UI: KPI bar with role selector, side-by-side Naive vs OpsGate cards, execution-trace pills, provenance badge, approval banner, MCP server list with add form, 3-tier memory inspector |
| `.streamlit/config.toml` | Navy and slate theme |
| `tests/test_pipeline.py`, `tests/test_optimizations.py` | Updated for new tool names, counts and the new ignorance text |

## Spec coverage
Spec steps 1 to 6 are implemented in code. Step 7 (RAGAS, Docker, CI) is not started; see `TODO.md`.

## Added in the second session
- `requirements.txt` (fastapi, uvicorn, httpx, rank-bm25, flashrank, sqlite checkpointer, qdrant, ragas, OTel, uv, ruff), `ruff.toml`, `Dockerfile`, `docker-compose.yml` (config validated, image not built), `.github/workflows/ci.yml`, `.env.example` v4 variables.
- `bench/eval_ragas.py` (RAGAS when keys exist, else a labelled heuristic) -> `bench/results_summary.md/.png`; `run_bench.py` prints token savings; `questions.json` out-of-scope expectation updated.
- README v4 section. One test bug fixed (`write_turn` default arg).
- Offline benchmark (20 questions, fake LLM, 21 tools): input tokens 1,933 -> 412 = **78.7% saved**, so the >85% target is NOT met offline; it must be measured with a real LLM. Quality numbers offline are not meaningful.

## Honest caveats
- Real Groq, Mem0, FlashRank and live weather / currency / web search were not tested here (no keys, no model download, network is restricted in this workspace).
- The Mem0 code is written against its documented API but is untested against a real Mem0 install.
- Offline fixture results (`UTILITY_OFFLINE=1`) are canned values, not live data.
