# HANDOFF: what is done, what is verified, what is left

Project: OpsGate Context Copilot. Authoritative design: `docs/BUILD_SPEC.md`. Do NOT restart from scratch; continue from this list.
Build environment had NO Groq key and NO Hugging Face access, so everything LLM-/embedding-dependent was only tested with offline fakes.

## Done and verified (offline)
| Phase | Files | Status |
|---|---|---|
| 0 Setup | `requirements.txt`, `.env.example`, `.gitignore`, `.vscode/`, `core/config.py`, `core/llm.py` (Groq factory + `FakeLLM` + `HashEmbedder` + `LocalEmbedder`) | Done |
| 1 Seed data | `scripts/seed_db.py` (500 orders, 200 noisy failed_txns with 25 cols), `scripts/make_runbooks.py` (2 PDFs + 1 HTML) | Done, counts verified |
| 2 MCP servers | `mcp_servers/` git (5), db (5), incident (5 + 2 resources + 2 prompts), runbook_rag (1) | Done; host lists 16 tools, resource + prompt calls verified |
| 3 Ingestion + RAG | `ingestion/*`, `core/rag_retriever.py` | Chunking verified (code and tables stay intact). Retrieval works but threshold NOT validated with a real embedder |
| 4 Deferred loading | `core/tool_registry.py` (hybrid 0.6 keyword + 0.4 cosine) | Verified: ops query binds search_tools + 2 tools |
| 5 Sandbox | `core/sandbox.py` | Verified: 200 rows, 134k chars -> 9.5k chars CSV + counts by service |
| 6 Cleaner | `core/cleaner.py` (7 steps) | Unit-tested |
| 7 Summarizer | `core/summarizer.py` | Implemented; only run through FakeLLM |
| 8 Memory | `memory/` (LocalStore offline, Mem0Store real), dedup, manager (bootstrap, episodic, end_session, forget, update) | LocalStore lifecycle verified incl. update-replaces-fact. **Mem0Store unverified** |
| 9 LangGraph | `core/graph.py` (`OpsGate`: bootstrap, router, chat, runbook, ops, naive, maybe_summarize, finalize; MemorySaver) | 6 offline tests pass |
| 10 Telemetry | `core/telemetry.py` (7-element token breakdown) | Done |
| 11 UI | `app.py` | Syntax-checked only, never launched |
| 12 Benchmark | `bench/questions.json` (16 q), `run_bench.py`, `plots.py` | Pipeline runs end to end offline; numbers meaningless (fakes) |
| 13 Docs | `README.md`, this file | Partial |

Tests: `EMBEDDING_PROVIDER=hash SIM_THRESHOLD=0.2 pytest -q tests` -> 6 passed.

## Optimisation pass (done, offline-verified: 26 tests pass)
KV-cache estimator + telemetry (`core/cache.py`), persistent MCP sessions (about 70x faster tool calls in a local timing: 0.04s vs 2.8s for 5 calls), `core/router.py` rules-first, sticky/ordered tools + host preload, cached Chroma client and embeddings, cleaner fixes (indentation, short-line dedupe), static reference in system prompt, memory dedupe cache + `add_raw` for `update()`, summary-time memory extraction, `SUMMARY_EVERY_N_TURNS`, 24-question benchmark with long/memory categories, new UI cards. Still unverified without keys: real Groq tool calling, Mem0Store (`add_raw` uses `infer=False`), threshold 0.45 with the real embedder, provider cache_read reporting.

## Second pass: LangSmith + detail views (done, 30 tests pass)
`core/tracing.py` (optional, redacted by default, hashed user id), telemetry detail logs, clickable Details popovers per card, `cap_tokens` + `MAX_TOOL_OUTPUT_TOKENS`, sandbox summary-first. NOT verified: a real LangSmith upload (no key here), and that redaction hides everything in your LangSmith version: open one trace and confirm no prompt text appears before trusting it.

## Third pass: chat threads + attached files (written WITHOUT being able to launch Streamlit, Chroma or LangGraph)
Files: `core/threads.py` (new), `core/uploads.py` (new), `tests/test_threads_uploads.py` (new), `app.py` (rewritten sidebar/input; telemetry cards kept), `core/graph.py` (`doc` route, `_rag_answer` helper, `doc_key`), `core/prompts.py` + `core/llm.py` (`UPLOAD_RULE`, `UPLOAD_IGNORANCE`; `source=` tag only on attached-file blocks; the system prompt and runbook blocks are unchanged), `ingestion/markdown_loader.py` (docx, tolerant text), `ingestion/contextual_indexer.py` (`get_collection(name=)`), `core/config.py` + `.env.example` (new settings), `requirements.txt` (`streamlit>=1.43`, `python-docx`), `.gitignore`, `README.md`.
Verified here: `core/threads.py` logic (titles, uniqueness, persistence, corrupt file) and `core/uploads.py` logic against a fake collection (per-thread isolation, replace on re-upload, bad type/size/empty errors, safe file names, overview sampling). NOT verified: real Chroma `where` filters, the Streamlit UI (paperclip `accept_file`, CSS classes `st-key-thr_<id>`, popover inside columns), LangGraph `doc` route, real PDFs through `pymupdf4llm`.
Known limits: thread LOG is saved to disk but the model's in-memory LangGraph history (`MemorySaver`) is lost on restart, so a restored thread shows the old messages while the model starts fresh (its attached files still work). Fix = a persistent checkpointer. Docx tables are appended after the text. Scanned PDFs are rejected (no OCR).

Engine policy for this pass: both models (OpsGate Context-Engineered and Naive Baseline) are kept and untouched. `core/graph.py` changed only by ADDING the `doc` route, the `doc_key` argument, and one line in `naive()` that also sees attached files; `runbook()`, `ops()`, `chat()` and the system prompt are byte-identical to v2. UI: light theme (`.streamlit/config.toml`), thread colours are soft yellow / red / green only (no blue).

## TODO (priority order)
0. **Run the third pass:** `pip install -U -r requirements.txt`, `pytest -q tests`, `streamlit run app.py`; attach a PDF, ask a question, check route `doc` in the card Details, check the thread title, rename/delete, switch threads. Fix whatever the first launch shows.
1. **Real-model run.** Put `GROQ_API_KEY` in `.env`, set `EMBEDDING_PROVIDER=local`, run `build_index.py`. Check: "Error 5042 retry script" returns the code chunk with score >= 0.45; "Lunar New Year policy" returns nothing. If not, tune `SIM_THRESHOLD` and record it.
2. **Verify Groq tool calling** with the adapters' tools in `OpsGate._tool_loop` (`llm.bind_tools(bound)` with MCP StructuredTools). Fix tool arg handling if the model omits `limit`.
3. **Verify Mem0Store** against the installed `mem0ai` (2.2.1 here): config keys (`custom_instructions` exists), `get_all`/`search` signatures, that results carry `[Semantic]/[Episodic]/[Procedural]` tags, and that `search` returns a `score`.
4. **Launch Streamlit** (`streamlit run app.py`) and fix UI issues.
5. **Benchmark:** add the missing categories (long conversation that triggers summarization, cross-session memory), run `run_bench.py`, run `plots.py`, put real numbers in README.
6. **Spec items not yet built:**
   - small static facts (SLA resource, service catalog) injected into the system prompt instead of RAG; the catalog resource is unused.
   - turn-based summary trigger (`should_summarize(turn_every=...)` exists but the graph does not pass it).
   - `end_session` is manual (UI button); also trigger extraction when the summary fires.
   - LLM-based router/classifier path was never run (only keyword rules offline).
7. Known small issues: `pymupdf4llm` splits `import time` into its own code chunk; `count_tokens` falls back to `len/4` when tiktoken cannot download its encoding; benchmark `correct` for chat only checks no tools/RAG.
8. README: architecture diagram, real benchmark table, 1-minute demo video.

## Rules (unchanged from the spec)
Never invent benchmark numbers; never clean user messages or history; documents go at the END of the prompt in XML tags; generate from original chunks only (never the enrichment summary); keep static prompt content stable (no timestamps).
