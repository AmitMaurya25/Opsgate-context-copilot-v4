"""LangGraph "Watchman" v4. mode = "naive" | "engineered".

engineered: bootstrap -> memory_recall -> router -> { chat | doc | rag_gate -> (hit) rag_answer | (miss) escalate_search -> agent <-> tools -> finish_tools }
            -> maybe_summarize -> memory_write (async) -> finalize
  Route A  RAG hit  (hybrid score >= 0.45)  -> grounded answer from [INTERNAL RUNBOOK]
  Route B  RAG miss -> deferred tool search over ALL connected MCP servers -> tools (RBAC + HITL gated) -> [LIVE MCP / WEB SEARCH]
  Route C  both empty -> "I cannot verify this from internal runbooks or live MCP tools."
naive:      bootstrap -> router -> naive_prep (ALL tool schemas, un-thresholded dense top-5, raw outputs) -> agent <-> tools -> finish_tools

run_comparison_turn() executes both pipelines at the same time for every query and reports tokens/latency deltas."""
import json, re, sqlite3, threading, time
from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import interrupt, Command
from core import config
from core.llm import get_llm, IGNORANCE, UPLOAD_IGNORANCE
from core.mcp_host import MCPHost, register_new_mcp_server
from core.tool_registry import ToolRegistry
from core.rag_retriever import search as rag_search
from core import uploads, guardrails
from core.cleaner import clean_context, cap_tokens
from core.tracing import setup_tracing, trace_config, audit, traced
from core.sandbox import run_programmatic_task
from core.summarizer import should_summarize, summarize, split_distal
from core.telemetry import Telemetry, count_tokens
from core.cache import PrefixCache, render_prompt
from core.router import route_by_rules
from core.prompts import build_system, build_user, SYSTEM_NAIVE, UPLOAD_RULE
from memory.manager import MemoryManager

TAG_RUNBOOK, TAG_LIVE = "[INTERNAL RUNBOOK]", "[LIVE MCP / WEB SEARCH]"

class S(TypedDict, total=False):
    question: str; mode: str; user_id: str; user_role: str; thread_id: str
    doc_key: str            # files attached to this thread ("" = none)
    messages: list; route: str; answer: str; tel: Telemetry
    loaded: list            # tools already loaded in this thread (sticky, most recent last, capped)
    memory: dict            # recalled tiers for THIS turn
    docs: list              # chunks that passed the RAG gate
    work: list              # working message list of the tool loop
    steps: int; tool_ok: bool; governed: bool; rag_failed: bool

setup_tracing()          # no-op unless LANGSMITH_API_KEY is set
MAX_STICKY_TOOLS = 4

def _useful(out: str) -> bool:
    """A tool result counts as evidence only if it is non-empty, not an error and not an empty collection."""
    o = (out or "").strip()
    if not o or o.lower().startswith(("tool error", "error executing tool", "guardrail", "access denied", "operator rejected", "skipped")): return False
    if o in ("[]", "{}", "null", "None"): return False
    try:
        j = json.loads(o)
        if isinstance(j, dict) and j.get("error"): return False
        if isinstance(j, (list, dict)) and not j: return False
    except Exception: pass
    return True

class OpsGate:
    def __init__(self, host: MCPHost | None = None, checkpoint_db=None):
        self.host = host or MCPHost().load()
        self._build_registry()
        self.llm = get_llm()
        self.memory: dict[str, MemoryManager] = {}
        self.pending: dict[str, dict] = {}                                      # thread_id -> naive result + timings while the engineered branch awaits approval
        self.cache = {"naive": PrefixCache(), "engineered": PrefixCache()}     # KV-cache hit estimate, one history per mode
        self.reference = self._load_reference()
        self._checkpoint_db = checkpoint_db
        self.graph = self._build()

    # ---------- helpers ----------
    def _build_registry(self):
        self.registry = ToolRegistry(self.host.tools, risk=self.host.risk, server_of=self.host.server_of)
        self.search_tool = self.registry.make_search_tool()
    def reload_tools(self):
        """Hot reload after mcp_servers.json changed: rebuild the registry; the graph reads self.registry at run time, so no graph change is needed."""
        self.host.reload(); self._build_registry(); self.reference = self._load_reference()
    def add_mcp_server(self, name: str, cfg: dict) -> dict:
        entry = register_new_mcp_server(name, cfg, host=self.host); self._build_registry(); self.reference = self._load_reference(); return entry
    def mem(self, uid):
        if uid not in self.memory: self.memory[uid] = MemoryManager(uid)
        return self.memory[uid]
    def _load_reference(self) -> str:
        """Small static docs (SLA policy, service catalog) go straight into the system prompt, not RAG: tiny, always relevant, identical for every user."""
        try:
            return "\n".join([self.host.read_resource("incident_ops", "service://sla-policy"), self.host.read_resource("incident_ops", "service://service-catalog")])
        except Exception: return ""
    def _call_llm(self, msgs, tel, tools=None):
        llm = self.llm.bind_tools(tools) if tools else self.llm
        tel.llm_calls += 1
        total, cached = self.cache.get(tel.mode, self.cache["engineered"]).observe(render_prompt(msgs, tools))
        if tel.llm_calls == 1:                                           # size of the static prefix (system prompt + tool schemas)
            tel.prefix_tokens = count_tokens(msgs[0].content if getattr(msgs[0], "type", "") == "system" else "") + \
                self.registry.schema_tokens([getattr(t, "name", "") for t in tools or [] if getattr(t, "name", "") in self.registry.entries])
        t0 = time.time(); resp = llm.invoke(msgs)
        tel.llm_log.append({"n": tel.llm_calls, "sent": total, "cached": cached, "ms": round(1000 * (time.time() - t0))})
        tel.record_call(total, cached, getattr(resp, "usage_metadata", None))
        tel.output_tokens += count_tokens(resp.content or "") + (count_tokens(json.dumps(resp.tool_calls, default=str)) if getattr(resp, "tool_calls", None) else 0)
        return resp
    def _account(self, tel, history, instructions):
        for m in history: tel.add("history", m.content)
        tel.add("instructions", instructions)
    @staticmethod
    def _cap(text, tel):
        text, dropped = cap_tokens(text, config.MAX_TOOL_OUTPUT_TOKENS); tel.truncated_tokens += dropped; return text
    def _run_tool(self, name, args, engineered, tel):
        try:
            schema = self.registry.entries[name]["schema"]["parameters"]
            paged = "page_size" in json.dumps(schema)
            if engineered and paged:
                out = run_programmatic_task(self.host, name, args)
                tel.cleaner_saved += max(0, (out["raw_chars"] - len(out["text"])) // 4)
                text = out["text"]
            else:
                res = self.host.call(name, args)
                raw = res if isinstance(res, str) else json.dumps(res, indent=2, default=str)
                if not engineered: return raw
                text = clean_context(raw); tel.cleaner_saved += max(0, (len(raw) - len(text)) // 4)
            text, findings = guardrails.sanitize_tool_output(text, self.registry.server(name))      # PII/secret redaction + injection quarantine BEFORE the LLM sees it
            if findings:
                tel.trace.append("Injection quarantined"); audit({"event": "prompt_injection_quarantined", "tool": name, "findings": findings})
            return self._cap(text, tel)
        except Exception as e:
            return f"tool error: {e}"
    def _bind(self, loaded, allowed=None):
        """search_tools first, then loaded tools in fixed registry order (stable KV prefix); a role never even sees tools it may not run."""
        names = [n for n in self.registry.ordered(loaded) if allowed is None or n in allowed]
        return [self.search_tool] + [self.host.tools[n] for n in names]

    # ---------- nodes ----------
    def bootstrap(self, s: S):
        tel = Telemetry(mode=s["mode"]); tel.role = guardrails.normalize_role(s.get("user_role"))
        msgs = s.get("messages") or []
        if msgs and msgs[0].type == "system": return {"tel": tel}
        if s["mode"] == "naive": return {"tel": tel, "messages": [SystemMessage(SYSTEM_NAIVE)]}
        return {"tel": tel, "messages": [SystemMessage(build_system([], [], self.reference))]}     # static prefix only; memory is recalled per turn

    def memory_recall(self, s: S):
        """Runs before routing on EVERY turn, whatever the thread: targeted search over Semantic, Episodic and Procedural memory."""
        if s["mode"] == "naive": return {"memory": {}}
        tel = s["tel"]
        try: rec = self.mem(s["user_id"]).recall(s["question"])
        except Exception: rec = {}
        tel.memory_used = {k: len(v) for k, v in rec.items()}
        tel.trace.append(f"Memory recall ({sum(tel.memory_used.values())})")
        return {"memory": rec}

    def router(self, s: S):
        tel = s["tel"]
        if s["mode"] == "naive": tel.route = "naive"; tel.route_reason = "naive"; return {"route": "naive"}
        q = s["question"]
        route = route_by_rules(q); tel.route_reason = "rule" if route else ""   # free and instant; keeps the first request prefix stable
        if route in (None, "runbook") and s.get("doc_key") and uploads.has_files(s["doc_key"]):
            tel.route = "doc"; tel.route_reason = "attached file"; return {"route": "doc"}
        route = route or "runbook"                                              # anything else goes to the RAG gate first, then self-heals via MCP
        tel.route_reason = tel.route_reason or "default"; tel.route = route; return {"route": route}

    def chat(self, s: S):
        tel, msgs = s["tel"], list(s["messages"]) + [HumanMessage(build_user(s["question"], None, s.get("memory")))]
        self._account(tel, msgs[1:-1], msgs[0].content); tel.add("prompt", s["question"]); tel.add("memory", json.dumps(s.get("memory") or {}))
        ans = self._call_llm(msgs, tel).content; tel.provenance = "chat"; tel.trace.append("Answer")
        return {"answer": ans, "messages": list(s["messages"]) + [HumanMessage(s["question"]), AIMessage(ans)]}

    def rag_gate(self, s: S):
        """Hybrid retrieval + rerank, then the >= 0.45 gate. Hit -> grounded answer. Miss -> escalate to MCP."""
        tel = s["tel"]; cands = rag_search(s["question"], k=5)
        best = cands[0]["score"] if cands else None; tel.top_score = best
        docs = [d for d in cands if d["score"] >= config.SIM_THRESHOLD]
        tel.chunks_kept = len(docs)
        tel.chunks = [{"section": d["section"], "kind": d["kind"], "score": d["score"], "tokens": count_tokens(d["text"])} for d in docs]
        tel.trace.append(f"RAG hit ({best:.2f})" if docs else (f"RAG miss ({best:.2f})" if best is not None else "RAG miss"))
        return {"docs": docs, "rag_failed": False}

    def rag_answer(self, s: S):
        """generate_grounded_answer: [INTERNAL RUNBOOK] context goes at the END of the prompt, ORIGINAL chunk text only."""
        tel, hist, docs = s["tel"], list(s["messages"]), [dict(d) for d in s["docs"]]
        for d in docs:
            raw = d["text"]; d["text"] = clean_context(raw) if d["kind"] == "prose" else raw   # lossless only on prose; keep code/tables byte-exact
            tel.cleaner_saved += max(0, (len(raw) - len(d["text"])) // 4)
        tel.contexts = [d["text"] for d in docs]
        user = build_user(s["question"], docs, s.get("memory"))
        self._account(tel, hist[1:], hist[0].content); tel.add("rag", "\n".join(d["text"] for d in docs)); tel.add("memory", json.dumps(s.get("memory") or {})); tel.add("prompt", s["question"])
        ans = self._call_llm(hist + [HumanMessage(user)], tel).content.strip()
        if ans.startswith(IGNORANCE[:30]): tel.trace.append("Runbook insufficient"); return {"rag_failed": True}   # the gate passed but the text did not answer: self-heal via MCP
        tel.provenance = "runbook"; tel.trace.append("Answer"); ans = ans if ans.startswith(TAG_RUNBOOK) else f"{TAG_RUNBOOK} {ans}"
        return {"answer": ans, "messages": hist + [HumanMessage(s["question"]), AIMessage(ans)]}

    def doc(self, s: S):
        """Question in a thread with attached files. Files first (even sample for 'summarise', similarity top-k otherwise); runbooks only as a fallback."""
        tel, hist, q, key = s["tel"], list(s["messages"]), s["question"], s["doc_key"]
        docs = uploads.overview_chunks(key) if uploads.is_overview(q) else uploads.retrieve_uploads(q, key)
        docs = docs or [d for d in rag_search(q, k=5) if d["score"] >= config.SIM_THRESHOLD]
        tel.chunks_kept = len(docs); tel.top_score = docs[0]["score"] if docs else None
        tel.chunks = [{"source": d.get("source"), "section": d["section"], "kind": d["kind"], "score": d["score"], "tokens": count_tokens(d["text"])} for d in docs]
        if not docs:
            tel.provenance = "ignorance"; return {"answer": UPLOAD_IGNORANCE, "messages": hist + [HumanMessage(q), AIMessage(UPLOAD_IGNORANCE)]}
        tel.contexts = [d["text"] for d in docs]; tel.trace.append("Attached file")
        user = UPLOAD_RULE + "\n\n" + build_user(q, docs, s.get("memory"))
        self._account(tel, hist[1:], hist[0].content); tel.add("rag", "\n".join(d["text"] for d in docs)); tel.add("memory", json.dumps(s.get("memory") or {})); tel.add("prompt", q)
        ans = self._call_llm(hist + [HumanMessage(user)], tel).content; tel.provenance = "upload"
        return {"answer": ans, "messages": hist + [HumanMessage(q), AIMessage(ans)]}

    def escalate_search(self, s: S):
        """deferred_tool_search_node: host-side search_tools across ALL connected MCP servers (RBAC-filtered), then the tool loop starts."""
        tel, hist, q = s["tel"], list(s["messages"]), s["question"]
        allowed = {n for n in self.registry.allowed(tel.role) if self.registry.server(n) != "runbook_rag"}   # the RAG gate already searched the runbooks
        found = self.registry.search_multi(q, config.ESCALATION_TOOLS, allowed)
        loaded = ([n for n in (s.get("loaded") or []) if n not in found] + found)[-MAX_STICKY_TOOLS:]
        tel.trace.append("Deferred search"); tel.route = "escalated" if s.get("route") != "ops" else "ops"
        self._account(tel, hist[1:], hist[0].content); tel.add("prompt", q); tel.add("memory", json.dumps(s.get("memory") or {}))
        return {"loaded": loaded, "work": hist + [HumanMessage(build_user(q, None, s.get("memory")))], "steps": 0, "tool_ok": False, "governed": False}

    def naive_prep(self, s: S):
        tel, hist, q = s["tel"], list(s["messages"]), s["question"]
        docs = rag_search(q, k=5, hybrid=False)                            # plain dense Top-K, no threshold
        if s.get("doc_key") and uploads.has_files(s["doc_key"]): docs = uploads.retrieve_uploads(q, s["doc_key"], threshold=0.0) + docs
        tel.chunks_kept = len(docs); tel.top_score = docs[0]["score"] if docs else None
        tel.chunks = [{"section": d["section"], "kind": d["kind"], "score": d["score"], "tokens": count_tokens(d["text"])} for d in docs]
        tel.contexts = [d["text"] for d in docs]; tel.trace.append("All tool schemas + top-5 RAG")
        self._account(tel, hist[1:], hist[0].content); tel.add("rag", json.dumps(docs, indent=2)); tel.add("prompt", q)
        return {"work": hist + [HumanMessage(build_user(q, docs))], "steps": 0, "loaded": [], "tool_ok": False, "governed": False}

    def agent(self, s: S):
        tel, eng = s["tel"], s["mode"] == "engineered"
        bound = self._bind(s.get("loaded"), self.registry.allowed(tel.role)) if eng else list(self.host.tools.values())
        names = [getattr(t, "name", "") for t in bound]
        tel.elements["tool_schemas"] = max(tel.elements["tool_schemas"], self.registry.schema_tokens([n for n in names if n in self.registry.entries]) + (150 if eng else 0))
        tel.tools_in_prompt = max(tel.tools_in_prompt, len(bound)); tel.total_tools = len(self.host.tools)
        resp = self._call_llm(s["work"], tel, bound)
        return {"work": s["work"] + [resp], "steps": s.get("steps", 0) + 1}

    def tools(self, s: S):
        """tool_execution_node. Order matters: (1) RBAC + human approval for every privileged call FIRST, with no side effect yet, because
        LangGraph re-runs this node on resume; (2) only then execute."""
        tel, eng, role = s["tel"], s["mode"] == "engineered", s["tel"].role
        calls = s["work"][-1].tool_calls
        verdict: dict = {}
        for tc in calls:
            name = tc["name"]
            if name == "search_tools" or name not in self.registry.entries: continue
            risk = self.registry.risk(name)
            if not eng:
                if guardrails.needs_approval(risk): verdict[tc["id"]] = "skipped"
            elif not guardrails.role_allows(role, risk): verdict[tc["id"]] = "denied"
            elif guardrails.needs_approval(risk):
                d = interrupt({"tool": name, "args": tc["args"], "reason": "Privileged write action requires operator approval", "risk": risk, "role": role})
                verdict[tc["id"]] = "approved" if (d or {}).get("approved") else "rejected"
        loaded, new, ok, governed = list(s.get("loaded") or []), [], s.get("tool_ok", False), s.get("governed", False)
        allowed = self.registry.allowed(role)
        for tc in calls:
            name, args, v = tc["name"], tc["args"], verdict.get(tc["id"])
            t0 = time.time(); status = "ok"
            if name == "search_tools":
                found = self.registry.search(args.get("query", ""), k=2, allowed=allowed if eng else None)
                loaded = ([x for x in loaded if x not in found] + found)[-MAX_STICKY_TOOLS:]; out = json.dumps(found); status = "search"
            elif eng and name not in loaded: out = "Guardrail: tool not loaded. Call search_tools first."; status = "blocked"
            elif v == "denied": out = f"Access denied: role {role} cannot run the PRIVILEGED_WRITE tool {name}."; status = "blocked"; governed = True
            elif v == "rejected": out = "Operator rejected this action. It was NOT executed."; status = "rejected"; governed = True
            elif v == "skipped": out = "Skipped: privileged write actions are only executed by the governed OpsGate pipeline."; status = "blocked"
            else:
                out = self._run_tool(name, args, eng, tel); status = "error" if out.startswith("tool error") else "ok"
                if status == "ok" and _useful(out): ok = True
                if self.registry.risk(name) == guardrails.PRIVILEGED_WRITE and v == "approved": governed = True
                tel.trace.append(name); tel.contexts.append(out[:1500])
            tel.log_tool(name, args, out, 1000 * (time.time() - t0), status)
            if name != "search_tools": audit({"event": "tool_call", "tool": name, "args": args, "status": status, "role": role, "mode": s["mode"], "risk": self.registry.risk(name), "approval": v or "n/a", "user": s.get("user_id"), "thread": s.get("thread_id")})
            tel.add("tool_io", out); new.append(ToolMessage(out, tool_call_id=tc["id"], name=name))
        return {"work": s["work"] + new, "loaded": loaded, "tool_ok": ok, "governed": governed}

    def finish_tools(self, s: S):
        """Route B/C resolution + provenance tag."""
        tel, hist, q, eng = s["tel"], list(s["messages"]), s["question"], s["mode"] == "engineered"
        last = s["work"][-1]
        text = (last.content if isinstance(last, AIMessage) and not last.tool_calls else "Stopped after too many tool steps.") or ""
        text = text.strip()
        if not eng: tel.provenance = "naive"; ans = text
        elif s.get("governed"): tel.provenance = "governance"; ans = text; tel.trace.append("Answer")
        elif not s.get("tool_ok") or text.startswith(IGNORANCE[:30]): tel.provenance = "ignorance"; ans = IGNORANCE; tel.trace.append("No evidence found")
        else: tel.provenance = "live_mcp"; ans = text if text.startswith(TAG_LIVE) else f"{TAG_LIVE} {text}"; tel.trace.append("Answer")
        return {"answer": ans, "messages": hist + [HumanMessage(q), AIMessage(ans)]}

    def maybe_summarize(self, s: S):
        if s["mode"] == "engineered" and should_summarize(s["messages"], config.SUMMARY_EVERY_N_TURNS or None):
            _, old, _ = split_distal(s["messages"])
            new, saved = summarize(s["messages"]); s["tel"].summary_saved = saved
            if old: self.mem(s["user_id"]).end_session(old)      # also extract memories at the summary trigger, before detail is lost
            return {"messages": new}
        return {}

    def memory_write(self, s: S):
        """Fire-and-forget AFTER the final answer: classify -> dedupe -> write to the matching tier. The user never waits on it."""
        if s["mode"] == "engineered":
            self.mem(s["user_id"]).write_async(s["question"], s.get("answer", ""), s.get("thread_id", ""), s["tel"].provenance)
        return {}

    def finalize(self, s: S):
        s["tel"].finish(); audit({"event": "turn", "mode": s["mode"], "role": s["tel"].role, "route": s["tel"].route, "provenance": s["tel"].provenance, "trace": s["tel"].trace,
                                  "tokens_in": s["tel"].sent_tokens, "latency_ms": round(s["tel"].latency_s * 1000), "user": s.get("user_id"), "thread": s.get("thread_id"), "question": s["question"][:300]})
        return {}

    # ---------- graph ----------
    def _checkpointer(self):
        serde = JsonPlusSerializer(allowed_msgpack_modules=[("core.telemetry", "Telemetry")])   # Telemetry rides in the checkpointed state: allow it explicitly
        path = self._checkpoint_db if self._checkpoint_db is not None else config.CHECKPOINT_DB
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            if str(path) != ":memory:": path = type(config.DATA)(path); path.parent.mkdir(parents=True, exist_ok=True)
            return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False), serde=serde)     # durable: an approval can arrive after a restart
        except Exception as e:
            print(f"[graph] SqliteSaver unavailable ({e!r}); using in-memory checkpointer"); return MemorySaver(serde=serde)

    def _build(self):
        g = StateGraph(S)
        for n in ["bootstrap", "memory_recall", "router", "chat", "rag_gate", "rag_answer", "doc", "escalate_search", "naive_prep", "agent", "tools", "finish_tools", "maybe_summarize", "memory_write", "finalize"]:
            g.add_node(n, traced(n)(getattr(self, n)))
        g.add_edge(START, "bootstrap"); g.add_edge("bootstrap", "memory_recall"); g.add_edge("memory_recall", "router")
        g.add_conditional_edges("router", lambda s: s["route"], {"chat": "chat", "static": "chat", "runbook": "rag_gate", "doc": "doc", "ops": "escalate_search", "naive": "naive_prep"})
        g.add_conditional_edges("rag_gate", lambda s: "rag_answer" if s["docs"] else "escalate_search", {"rag_answer": "rag_answer", "escalate_search": "escalate_search"})
        g.add_conditional_edges("rag_answer", lambda s: "escalate_search" if s.get("rag_failed") else "maybe_summarize", {"escalate_search": "escalate_search", "maybe_summarize": "maybe_summarize"})
        g.add_edge("escalate_search", "agent"); g.add_edge("naive_prep", "agent")
        g.add_conditional_edges("agent", lambda s: "tools" if getattr(s["work"][-1], "tool_calls", None) and s["steps"] < config.MAX_AGENT_STEPS else "finish_tools", {"tools": "tools", "finish_tools": "finish_tools"})
        g.add_edge("tools", "agent")
        for n in ["chat", "doc", "finish_tools"]: g.add_edge(n, "maybe_summarize")
        g.add_edge("maybe_summarize", "memory_write"); g.add_edge("memory_write", "finalize"); g.add_edge("finalize", END)
        return g.compile(checkpointer=self._checkpointer())

    # ---------- public API ----------
    def _cfg(self, user_id, mode, thread_id):
        cfg = {"configurable": {"thread_id": thread_id or f"{user_id}-{mode}"}}
        cfg.update(trace_config(mode, user_id, cfg["configurable"]["thread_id"])); return cfg
    def _result(self, cfg, out) -> dict:
        st = self.graph.get_state(cfg)
        if st.next:                                                      # paused by interrupt(): waiting for an operator decision
            payloads = [i.value for t in st.tasks for i in t.interrupts]
            tel = st.values.get("tel")
            return {"status": "interrupted", "answer": "", "provenance": "", "interrupt_payload": payloads[0] if payloads else {}, "telemetry": tel.as_dict() if tel else {}}
        return {"status": "ok", "answer": out["answer"], "provenance": out["tel"].provenance, "interrupt_payload": None, "telemetry": out["tel"].as_dict()}
    def ask(self, question: str, mode: str = "engineered", user_id: str = "demo", thread_id: str | None = None, doc_key: str = "", user_role: str | None = None) -> dict:
        cfg = self._cfg(user_id, mode, thread_id)
        out = self.graph.invoke({"question": question, "mode": mode, "user_id": user_id, "doc_key": doc_key or "", "user_role": user_role or config.DEFAULT_ROLE,
                                 "thread_id": cfg["configurable"]["thread_id"]}, cfg)
        return self._result(cfg, out)
    def resume(self, thread_id: str, approved: bool, mode: str = "engineered", user_id: str = "demo") -> dict:
        """Continue a paused thread: Command(resume={"approved": ...})."""
        cfg = self._cfg(user_id, mode, thread_id)
        out = self.graph.invoke(Command(resume={"approved": bool(approved)}), cfg)
        return self._result(cfg, out)
    def end_session(self, user_id: str, mode: str = "engineered", thread_id: str | None = None):
        cfg = {"configurable": {"thread_id": thread_id or f"{user_id}-{mode}"}}
        st = self.graph.get_state(cfg).values
        return self.mem(user_id).end_session(st.get("messages", [])) if st else []

    # ---------- dual pipeline ----------
    @staticmethod
    def _branch(r: dict, wall_ms: float, mode: str, total_tools: int) -> dict:
        t = r.get("telemetry") or {}
        b = {"answer": r["answer"], "input_tokens": t.get("sent_tokens", 0), "output_tokens": t.get("output_tokens", 0), "latency_ms": round(wall_ms), "llm_calls": t.get("llm_calls", 0),
             "context_tokens": t.get("input_tokens", 0), "provenance": r.get("provenance", ""), "telemetry": t}
        if mode == "naive": b.update(naive_input_tokens=b["input_tokens"], naive_output_tokens=b["output_tokens"], naive_latency_ms=b["latency_ms"], tools_injected=t.get("tools_in_prompt", 0), total_tools=total_tools)
        else: b.update(opsgate_input_tokens=b["input_tokens"], opsgate_output_tokens=b["output_tokens"], opsgate_latency_ms=b["latency_ms"], tools_loaded=t.get("tools_in_prompt", 0), total_tools=total_tools, execution_trace=t.get("trace", []), memory_used=t.get("memory_used", {}))
        return b
    def _compare(self, naive: dict, ops: dict, thread_id: str, query: str, role: str, status: str, payload) -> dict:
        n, o = naive, ops
        saved = n["input_tokens"] - o["input_tokens"]
        return {"status": status, "thread_id": thread_id, "query": query, "user_role": role, "interrupt_payload": payload, "naive": n, "opsgate": o,
                "delta": {"tokens_saved": saved, "savings_pct": round(saved / max(n["input_tokens"], 1) * 100, 1), "latency_saved_ms": n["latency_ms"] - o["latency_ms"]}}
    def run_comparison_turn(self, user_query: str, thread_id: str, user_id: str = "demo", user_role: str | None = None, doc_key: str = "") -> dict:
        """Both pipelines, simultaneously, for the SAME query. If the engineered branch needs approval the result is status="interrupted";
        call resume_comparison() with the operator's decision."""
        role = guardrails.normalize_role(user_role or config.DEFAULT_ROLE)
        def run(mode):
            t0 = time.perf_counter(); r = self.ask(user_query, mode, user_id, f"{thread_id}-{mode}", doc_key, role); return r, (time.perf_counter() - t0) * 1000
        with ThreadPoolExecutor(max_workers=2) as ex:
            fn, fo = ex.submit(run, "naive"), ex.submit(run, "engineered")
            (rn, wn), (ro, wo) = fn.result(), fo.result()
        total = len(self.host.tools)
        naive = self._branch(rn, wn, "naive", total)
        if ro["status"] == "interrupted":
            ops = self._branch(ro, wo, "engineered", total)
            self.pending[thread_id] = {"naive": naive, "ops_ms": wo, "query": user_query, "role": role, "user_id": user_id}
            return self._compare(naive, ops, thread_id, user_query, role, "interrupted", ro["interrupt_payload"])
        return self._compare(naive, self._branch(ro, wo, "engineered", total), thread_id, user_query, role, "ok", None)
    def resume_comparison(self, thread_id: str, approved: bool) -> dict:
        p = self.pending.get(thread_id)
        if not p: raise KeyError(f"no pending approval for thread {thread_id}")
        t0 = time.perf_counter(); r = self.resume(f"{thread_id}-engineered", approved, "engineered", p["user_id"]); p["ops_ms"] += (time.perf_counter() - t0) * 1000   # operator think-time is not counted
        total = len(self.host.tools); ops = self._branch(r, p["ops_ms"], "engineered", total)
        if r["status"] == "interrupted": return self._compare(p["naive"], ops, thread_id, p["query"], p["role"], "interrupted", r["interrupt_payload"])
        self.pending.pop(thread_id, None)
        return self._compare(p["naive"], ops, thread_id, p["query"], p["role"], "ok", None)
