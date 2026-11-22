# OpsGate Context Copilot: Full Build Spec (one-go, for a coding agent)

## 0. Rules for the agent
- Build **phase by phase** (Section 6). Run each phase's check before moving on. Commit after each phase.
- No paid APIs except the Groq LLM (free tier). Everything else is local.
- Never invent benchmark numbers. README figures must come from `bench/results.csv`.
- Keep it small: 3 MCP servers x 5 tools, 1 RAG server, 2 runbook PDFs, 24 benchmark questions.
- If a library import or API differs from this spec, read the installed version's docs. Do not guess.

## 1. Goal
A LangGraph agent connected to several MCP servers that proves **context engineering**. A Streamlit UI toggles two modes and shows the difference live.

| Mode | Behaviour |
|---|---|
| `naive` | All tool schemas loaded, Top-5 RAG with no threshold, raw JSON tool output, no summarization, no memory structure |
| `engineered` | Watchman layer: router, deferred loading, pagination, thresholded RAG, lossless cleaning, lossy summary, structured memory, cache-friendly prompt layout |

**Success:** engineered mode uses clearly fewer input tokens than naive at equal or better correctness, refuses out-of-scope questions with zero hallucination, and the UI shows a per-element token breakdown.

## 2. Concept coverage (every study topic maps to a feature)
| Study topic | Where it is built |
|---|---|
| Why context engineering; 7 context elements (history, prompt, instructions, tool schemas, tool I/O, RAG, memory); stateless LLMs | `core/telemetry.py` breaks tokens down by these 7 elements; UI shows the breakdown |
| Tools and tool calling cycle | LangGraph tool loop; MCP tools are plain Python functions |
| MCP basics: Host, Server, handshake, tools/resources/prompts | Host = LangGraph app; 4 FastMCP servers; startup discovery |
| MCP as a context problem; "Watchman" filtration layer | `core/` pipeline sits between MCP/RAG output and the LLM |
| Deferred loading | `core/tool_registry.py` (`search_tools`) |
| When NOT to defer (5 to 6 tools) | Config flag `DEFER_MIN_TOOLS=8`: below it, bind tools directly |
| Programmatic tool calling (when to use / not) | `core/sandbox.py` runs multi-call loops outside the LLM; used only for many-call or large-data tasks |
| Tool output problems: design better tools, pagination, keep structure, summarize | Tools return only needed fields, `page`/`page_size` args, CSV tables, summary node |
| Lossless compression (7 techniques, `clean_context`) | `core/cleaner.py` |
| Lossy compression: triggers, segmentation, secondary LLM, 5-point state summary | `core/summarizer.py` |
| Hybrid compression framework (lossless first, lossy second) | Order enforced in `graph.py` |
| Documents: markdown-aware loaders vs flat parsers; HTML to Markdown | `ingestion/markdown_loader.py` (PDF and HTML) |
| Chunking pitfalls; 3-way triage (prose/code/table); Lost in the Middle | `ingestion/triage_splitter.py` |
| Contextual retrieval ("amnesia" fix) | `ingestion/contextual_indexer.py` |
| Embeddings, vector DB, cosine retrieval, same model for query, N=5 | `core/rag_retriever.py` with Chroma |
| Separation of concerns, ignorance clause, context at end | `core/prompts.py` |
| RAG vs direct context (small data: direct + KV cache) | Router rule: small static docs go in the system prompt, not RAG |
| Similarity threshold 0.45 instead of plain Top-K | `rag_retriever.py` and episodic memory |
| Three-case agentic logic (knowledge / action / chat) | `router` node in `graph.py` |
| KV caching: static on top, dynamic at bottom, avoid cache breaking | Prompt layout rules in `core/prompts.py` (Section 5) |
| Writing system instructions (persona/task, constraints, format, audience) | `core/prompts.py` system prompt |
| Delimiter prompting (XML tags) | All dynamic blocks wrapped in XML tags |
| KV-cache hit tracking (measure, not just design) | `core/cache.py` `PrefixCache` + `telemetry.py` (`sent_tokens`, `cached_tokens`, `cache_hit_pct`, provider `cache_read` when reported) |
| Memory: semantic / episodic / procedural; segregation; dict vs list mapping | `memory/` package |
| Mem0 library: add/search, user isolation, config | `memory/mem0_store.py` |
| Memory limits and hybrid engineering (forget/update wrappers, static facts in system prompt) | `memory/manager.py` |
| Four-stage memory lifecycle: extract, store, retrieve, inject | `memory/manager.py` plus `graph.py` bootstrap |

## 3. Stack and install
Python 3.11+, virtual environment.
```bash
python -m venv .venv && source .venv/bin/activate
pip install mcp langchain-mcp-adapters langgraph langchain langchain-groq langchain-text-splitters \
  chromadb pymupdf4llm sentence-transformers tiktoken mem0ai streamlit pandas matplotlib \
  reportlab python-dotenv beautifulsoup4 markdownify
pip freeze > requirements.txt
```
`.env.example`:
```
GROQ_API_KEY=
GROQ_MODEL=llama-3.3-70b-versatile
SUMMARIZER_MODEL=llama-3.1-8b-instant      # small, cheap secondary LLM
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
SIM_THRESHOLD=0.45
SUMMARY_TRIGGER_TOKENS=1200                # demo value; real systems use 30000+
SUMMARY_KEEP_LAST_N=2
DEFER_MIN_TOOLS=8
PAGE_SIZE=50
SUMMARY_EVERY_N_TURNS=0                    # 0 = token trigger only
CONTEXT_BUDGET_TOKENS=8000                 # UI budget bar
AUTO_PRELOAD_TOOLS=1                       # host-side search_tools, saves one LLM round trip
CACHE_BLOCK_TOKENS=64                      # KV-cache block size for the hit-rate estimator
PERSISTENT_MCP=1                           # keep MCP server processes alive between calls
MAX_TOOL_OUTPUT_TOKENS=3000                # hard cap on any tool result entering the prompt
LANGSMITH_API_KEY=                         # optional; empty = tracing off. Key lives ONLY in .env
LANGSMITH_PROJECT=opsgate-context-copilot
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_REDACT=1                         # 1 = prompts / tool I/O are not sent to LangSmith
```
Mem0 needs an LLM and embedder in its config. Point it at Groq plus the same local embedder, with a local vector store (Qdrant local path or Chroma). The vector dimension must match the embedder.

## 4. Directory structure
```
opsgate-context-copilot/
├── README.md  requirements.txt  .env.example  .gitignore
├── data/ runbooks/  mock_ops.db  chroma/  mem0/  (chroma, mem0 gitignored)
├── scripts/ seed_db.py  make_runbooks.py  build_index.py
├── mcp_servers/
│   ├── git_ops_server.py  db_logs_server.py  incident_server.py
│   └── runbook_rag_server.py
├── ingestion/ markdown_loader.py  triage_splitter.py  contextual_indexer.py
├── memory/ mem0_store.py  manager.py  dedup.py
├── core/
│   ├── config.py  prompts.py  telemetry.py
│   ├── tool_registry.py  sandbox.py  cleaner.py  summarizer.py  router.py  cache.py  mcp_host.py  llm.py
│   ├── rag_retriever.py  graph.py
├── app.py
├── bench/ questions.json  run_bench.py  results.csv  plots.py
└── tests/ (one smoke test per phase)
```

## 5. Key design rules

**Prompt layout (KV caching + separation of concerns).** Build every prompt in this exact order. Never put anything that changes per request above the dynamic section, and never put timestamps or random IDs in the static part (that breaks the cache from that point on).
```
[STATIC TOP]    system instruction (persona/task, constraints, style/format, audience)
                + semantic facts + procedural rules (cold-loaded memory)
                + tool schemas (only loaded ones, stable order)
[DYNAMIC BOTTOM] conversation summary note (if any)
                 + recent messages
                 + <episodic_memory>...</episodic_memory>   (only hits >= threshold)
                 + <documents><document id="1">...</document></documents>  (RAG, at the END)
                 + <question>...</question>
```
Formula: `New System Instruction = Old System Instruction + Rules (procedural) + Facts (semantic)`.

**System instruction content.** Persona and task, constraints (including the ignorance clause: "If the documents do not contain the answer, say: I cannot answer this based on the provided runbooks."), style and format (compact tables), audience (on-call engineers). Write it at the right altitude: specific enough to guide, not a rigid script. Start small and iterate.

**Compression order (hybrid framework).** Lossless first, lossy second. The cleaning boundary sits between retrieval (MCP and RAG) and the context window. **Never clean or alter the user prompt or message history.** Never summarize system instructions or tool definitions.

**Three-case router.** `chat` (no tools, no RAG), `runbook` (knowledge/policy, RAG), `ops` (actions/live data, deferred tools). Small static facts (for example the SLA policy resource) go straight into the system prompt instead of RAG.

**KV-cache rules (hard requirements).**
1. Order: `[system base] → [<reference> static docs] → [procedural rules] → [semantic facts] → [tool schemas] → [summary note] → [history] → [<episodic_memory>] → [<documents>] → [<question>]`. Parts shared by every user come first, per-user parts later, per-request parts last.
2. Byte-stable static part: no timestamps or random IDs; memory lists are de-duplicated, `sorted()`, and the merge result is memoised by a hash of the records (the LLM merge must not re-run and re-word the prompt).
3. Tool schemas are always bound in fixed registry order, never score order. The loaded set is sticky per thread (capped at 4, most recent kept), so it only grows or shifts slowly instead of flipping each turn.
4. History keeps the plain question, not the retrieved documents of that turn (saves tokens; the cache is shared up to the previous turn).
5. Summaries rebuild only the tail: `[system + tools] + [summary note] + [last N]`. System prompt and tools never change, so the prefix survives summarisation.
6. Measure it: log `sent_tokens` (all LLM calls of a run, since tool loops re-send the context), `cached_tokens` (provider `usage_metadata.input_token_details.cache_read` when present, else prefix estimate) and `effective_tokens = sent - cached`. Show the hit rate in the UI.

**Small static docs.** `service://sla-policy` and `service://service-catalog` are read once at startup and placed in `<reference>` inside the system prompt. Router label `static` answers them with no tools and no RAG.

**Router.** Free regex rules first (greeting, static, ops, runbook); the LLM classifier runs only if no rule matches.

**Host performance.** One background event loop keeps every MCP server process and session alive (`PERSISTENT_MCP=1`), with automatic fallback to per-call processes. Cache the Chroma client and query embeddings. Cache token counts.

**Observability (LangSmith + local detail).** `core/tracing.py` enables LangSmith only when `LANGSMITH_API_KEY` is set. Default `LANGSMITH_REDACT=1` hides inputs and outputs, so only run structure, order and timings leave the machine; user ids are hashed. Tracing never touches the prompt (KV-cache layout unchanged). Telemetry also keeps local-only detail: `tool_calls` (order, tool, args, ms, output tokens, status), `llm_log` (sent/cached/ms per call), `chunks` (section, kind, score, tokens), `route_reason`. In the UI each card has a **Details** popover showing its breakdown.

**No reckless data into the LLM.** Every tool result in engineered mode is cleaned, then hard-capped at `MAX_TOOL_OUTPUT_TOKENS` on a line boundary with a truncation note. The sandbox puts the answer-bearing summary line (`counts by service`) before the CSV so truncation can never drop it.

**Cleaner safety.** Whitespace collapse must never touch leading indentation (code, diffs). Line de-duplication ignores lines shorter than 8 characters (`}`, `end`, `return`).

## 6. Phases

### Phase 0: Setup
Repo, venv, `.env`, `core/config.py`, `.gitignore`. **Check:** config imports and a one-line Groq call returns text.

### Phase 1: Seed data
- `seed_db.py`: SQLite with `orders` (500 rows), `failed_txns` (200 rows: `service`, `error_code`, `ts`, `trace_id`, `internal_uuid` and about 20 more metadata fields to create noisy output), `latency_stats`.
- `make_runbooks.py`: 2 PDFs with prose, a Python retry script, and an error-code table (include **Error 5042**, payment worker). Also one small HTML runbook page to test the HTML loader.
- Mock git data JSON (commits, branches, PRs) for 3 services.
**Check:** row counts match; PDFs contain a code block and a table.

### Phase 2: MCP servers
Use `FastMCP`, stdio transport. Tool list is in Section 7. Tools must have clear docstrings and typed args. Some DB tools return bulky raw JSON on purpose. Add 2 resources and 2 prompts.
**Check:** `MultiServerMCPClient` lists 16 tools and calls one tool per server.

### Phase 3: Ingestion and RAG
1. `markdown_loader.py`: PDF via `pymupdf4llm` (keeps headings, code, tables). HTML via `beautifulsoup4` plus `markdownify`, dropping scripts, nav and footers. Do not use flat text extractors.
2. `triage_splitter.py`: scan line by line, tag blocks `prose`, `code`, `table`, carry `chapter` and `section` from headings. Prose: `RecursiveCharacterTextSplitter` (500, overlap 60). Code: Python code splitter (1500). Tables: kept whole.
3. `contextual_indexer.py`: LLM writes a 1 to 2 sentence summary situating each chunk in its document. **Embed summary + chunk.** Store the **original chunk only** as the Chroma document; summary and `chapter`, `section`, `kind` go in metadata. Collection uses cosine space.
4. `rag_retriever.py`: embed the query with the **same model**, fetch N=5, similarity = `1 - distance`, drop everything under `SIM_THRESHOLD`, return original chunks with scores and sections. Empty result is valid.
5. `runbook_rag_server.py`: tool `search_runbooks(query, k=5)`.
**Check:** "Error 5042 retry script" returns the code chunk with indentation intact and score at least 0.45. "Lunar New Year policy" returns empty. If 0.45 does not separate these for your embedder, tune it and record the final value.

### Phase 4: Tool registry (deferred loading)
- Startup: connect to all servers, fetch every tool, embed name plus description, keep schemas in a host-side dictionary.
- `search_tools(query, k=2)`: score = `0.6 * keyword_overlap + 0.4 * cosine`.
- Engineered mode: if total tools is at least `DEFER_MIN_TOOLS`, the LLM sees only `search_tools` first; after it returns, bind only the matched schemas for the next call. Otherwise bind directly.
- Block calls to tools that are not loaded.
- Bind in registry order; sticky loaded set per thread (max 4); optional host-side preload (`AUTO_PRELOAD_TOOLS`).
**Check:** a failed-orders query loads at most 2 tool schemas.

### Phase 5: Sandbox, pagination, output curation
- All list tools accept `page` and `page_size`, and return `{rows, page, total, more_pages_available}`.
- `sandbox.run_paged(tool, args, page_size, max_rows)` loops pages **outside the LLM** (programmatic tool calling) and returns one final result.
- `strip_fields(rows, keep)`, `to_csv(rows)` (header once, no repeated keys). Use programmatic mode only for many-call or large-data tasks; a single simple call uses plain tool calling.
**Check:** log raw vs returned token counts for 200 orders; returned must be far smaller.

### Phase 6: Lossless cleaner (`core/cleaner.py`)
`clean_context(text)` is a sequential pipeline: (1) strip ANSI codes, (2) strip HTML/XML tags but keep inner values, (3) remove standard timestamps, (4) remove decorative characters (ASCII borders, dashes), (5) collapse whitespace and blank lines, (6) minify JSON, (7) dedupe repeated rows or blocks. JSON to CSV conversion happens in the sandbox. Apply to MCP and RAG output only.
**Check:** unit tests prove facts (IDs, numbers, error codes) survive; report percentage saved per category in a test printout.

### Phase 7: Lossy summarizer (`core/summarizer.py`)
- **Trigger:** token count over `SUMMARY_TRIGGER_TOKENS`, or every N turns (configurable).
- **Segment:** never touch system instructions or tool definitions; exclude the last `SUMMARY_KEEP_LAST_N` messages.
- **Summarize** the distal history with the small `SUMMARIZER_MODEL`. The summary must contain five state-tracking parts: established decisions, established facts, user preferences, unsolved/pending tasks, main goal and progress.
- **Rebuild:** `[system + tools] + [summary note] + [last N messages]`.
- Deletion (sliding window) is not used because it can drop foundational facts.
**Check:** a long scripted chat drops token load after the trigger and the agent still remembers earlier facts (do not repeat questions already answered).

### Phase 8: Memory (`memory/`)
- **Lifecycle:** extract, store, retrieve, inject.
- **Extract and store:** Mem0 `add()` with custom instructions that tag every fact `[Semantic]`, `[Episodic]` or `[Procedural]` (dual tags allowed), normalize first person to third person, and ignore noise (for example "what is the capital of France"). Run extraction **after the session** and also at the summary trigger. Isolate by `user_id`.
- **Segregate:** semantic and procedural into dictionaries with upsert (one true answer per key); episodic stays an append-only list retrieved by search.
- **De-duplicate (`dedup.py`):** a small LLM merges conflicts in the semantic and procedural lists, keeps the most recent by timestamp, prunes outdated facts. Add a `forget(key)` and `update(key, value)` wrapper (Mem0 does not replace facts automatically). Make historical-keeping a config option.
- **Retrieve and inject:** at session start (bootstrap), cold-load semantic and procedural lists into the **static system prompt** (no retrieval, so the cache stays stable). For each query, run Mem0 `search()`, keep only `[Episodic]` hits at or above 0.45 (top 2), and inject into the **user section** inside `<episodic_memory>` tags.
- Hybrid rule: very frequent identity facts live in the system prompt, not behind retrieval.
**Check:** tell the agent "I am vegetarian and I prefer Python" in session 1; in a new session ask for a snack and a script; both preferences are applied without being repeated. Updating a fact replaces the old one.

### Phase 9: LangGraph (`core/graph.py`)
Nodes: `bootstrap` (load memory), `router`, `chat`, `runbook_rag`, `ops_agent`, `clean`, `summarize`, `finalize`, `post_session_memory`. Use `MemorySaver`.
- `runbook_rag`: retrieve, `clean`, build the layout from Section 5; empty result returns the ignorance phrase exactly.
- `ops_agent`: deferred tools, sandbox, `clean`, answer.
- `naive` mode skips the router, cleaner, summarizer and memory structure.
- Every run records telemetry.
**Check:** "Hi, who are you?" uses no tools and no RAG in engineered mode.

### Phase 10: Telemetry
`telemetry.py` records per run: `sent_tokens`, `cached_tokens`, `cache_hit_pct`, `effective_tokens`, static prefix size, total input tokens (provider usage when available, `tiktoken` fallback), **tokens per context element** (history, prompt, instructions, tool schemas, tool I/O, RAG, memory), tools in prompt vs total, top similarity score, chunks kept, tokens saved by cleaner and summarizer, and latency.

### Phase 11: Streamlit UI (`app.py`)
Sidebar: mode switch, user ID, "end session" (triggers memory extraction), clear chat, loaded-memory panel. After every answer: 4 cards (context tokens with live delta vs the other mode for the same prompt, tools loaded n/total, retrieval status, KV-cache hit) plus a stacked bar of the 7 context elements and a context-budget bar.
**Check:** the same prompt in both modes shows clearly different numbers.

### Phase 12: Benchmark
`questions.json`: 24 questions, 4 per category: chat, runbook, out-of-scope, ops, long-conversation (summary), memory (cross-session). Each has an automatic `expected` check. `run_bench.py` runs both modes (multi-turn `setup` for long, two sessions for memory, `update`/`forbid` checks) and writes `results.csv` with `question_id, category, mode, input_tokens, tools_in_prompt, rag_tokens, tool_output_tokens, correct, latency_s, sent_tokens, cached_tokens, cache_hit_pct, effective_tokens, summary_saved, route`. `plots.py` draws average tokens and accuracy per mode and per category.
**Check:** 48 rows and charts render.

### Phase 13: README and demo
Problem, architecture diagram, mode comparison, setup steps, real benchmark table and charts, the 5 demo prompts, limitations, 1-minute demo video.

## 7. MCP server specs
All tools have docstrings (the registry searches them) and JSON-serializable output.
- **git_ops_server (5):** `list_branches(service)`, `get_recent_commits(service, limit=10)`, `get_commit_diff(service, sha)`, `list_open_prs(service, page=1, page_size=50)`, `blame_file(service, path)`.
- **db_logs_server (5):** `query_failed_orders(limit=50, page=1, page_size=50)` (bulky raw JSON), `get_order(order_id)`, `count_errors_by_service(hours=24)`, `get_latency_stats(service)`, `describe_schema()`.
- **incident_server (5):** `list_active_alerts()`, `get_oncall(service)`, `trigger_rollback(service, version)` (mock confirmation only), `draft_incident_email(service, summary)`, `get_sla_status(service)`.
- **runbook_rag_server (1):** `search_runbooks(query, k=5)` returns `[{text, score, section}]` after the threshold.
- **Resources:** `service://sla-policy`, `service://service-catalog`. **Prompts:** `postmortem_template`, `incident_summary`.

## 8. Demo prompts
1. "Hi, who are you?" shows no tools, no RAG, tiny token use.
2. "What is our policy on Lunar New Year holidays?" shows empty retrieval and the ignorance clause.
3. "How do I fix Error 5042 in the payment worker? Show the Python retry script." shows intact table and code, answer from raw chunks with section citation.
4. "Check the last 200 failed orders, find which service threw errors, and show its latest commit." shows deferred loading, pagination, cleaned CSV and a root-cause answer.
5. Session 2: "Suggest a quick fix script." shows remembered preferences (Python, step by step) loaded from memory.

## 9. Pitfalls
- Do not show the enrichment summary to the generator; answer from original chunks only.
- Do not mix up cosine distance and similarity.
- Do not reset the stable system prompt order between turns, and keep tool schema order stable.
- Do not clean user messages or history.
- Do not store memory without tags or user isolation.
- Do not publish token-saving claims before Phase 12.
- Never commit `.env` or the vector store folders.
- Do not publish `results.csv` produced with the offline fake LLM or hash embedder.
- Do not put a timestamp, user name or retrieved text above the static prefix.
- Do not reorder tools by relevance score; do not re-run the memory merge LLM for unchanged memories.
- Do not collapse leading whitespace in tool output (breaks code and diffs).

## 10. Definition of done
1. All phase checks pass.
2. `streamlit run app.py` works from a clean clone using README steps only.
3. `bench/results.csv` exists and the README numbers match it.
4. Out-of-scope questions return the ignorance phrase every time.
5. Cross-session memory demo works, including an updated (not duplicated) fact.
6. Repo is public with README, diagram and demo link.
7. KV-cache hit rate is visible in the UI and in `results.csv`; second-turn hit rate in engineered mode is clearly above naive.
8. `pytest -q tests` (offline) passes, including `tests/test_optimizations.py`.
9. With a LangSmith key, one run shows nodes and tool calls with timings and NO prompt text (redaction on); without a key the app runs unchanged.
10. Each UI card opens a Details view (token elements, tool calls, chunks and scores, per-LLM-call cache).
