"""OpsGate Context Copilot: chat UI. Every message runs BOTH pipelines (Naive baseline vs OpsGate context-engineered) and shows them side by side.
Thin client: talks to the FastAPI backend when OPSGATE_API_URL is set (docker-compose does), otherwise runs the same code in-process."""
import html, os, re
import streamlit as st
from core import config
from core.client import get_client
from core.guardrails import ROLES
from core.threads import ThreadStore, bucket
from core.telemetry import ELEMENTS

st.set_page_config(page_title="OpsGate Context Copilot", layout="wide", page_icon=None)
NAVY, GREEN, RED, AMBER, SLATE = "#12284c", "#1e8e5a", "#c0392b", "#b7791f", "#5b6b82"
st.markdown(f"""<style>
.block-container{{padding-top:1.4rem;max-width:1500px}}
.og-top{{display:flex;align-items:center;justify-content:space-between;margin-bottom:.4rem}}
.og-top h1{{font-size:1.35rem;margin:0;color:{NAVY};letter-spacing:-.01em}} .og-top span{{color:{SLATE};font-size:.85rem}}
.kpi{{background:#fff;border:1px solid #e3e8ef;border-radius:10px;padding:.7rem .95rem}}
.kpi small{{color:{SLATE};font-size:.75rem;display:block}} .kpi b{{font-size:1.45rem;color:{NAVY}}}
.card{{background:#fff;border:1px solid #e3e8ef;border-radius:10px;padding:.9rem 1rem;height:100%}}
.card.naive{{border-top:4px solid {RED}}} .card.ops{{border-top:4px solid {GREEN}}}
.card h4{{margin:0 0 .5rem;font-size:1rem;color:{NAVY}}}
.badge{{display:inline-block;font-size:.72rem;padding:.12rem .55rem;border-radius:999px;margin:0 .3rem .35rem 0;font-weight:600}}
.b-red{{background:#fdecea;color:#9b2c20}} .b-amber{{background:#fdf3dd;color:#8a5a12}} .b-green{{background:#e3f5ec;color:#16694a}}
.b-blue{{background:#e6eefb;color:#1f4e9c}} .b-grey{{background:#edf0f5;color:#44546a}}
.trace{{margin:.2rem 0 .6rem;font-size:.74rem}} .trace .p{{display:inline-block;background:#edf0f5;color:#2b3a52;border-radius:6px;padding:.1rem .5rem}}
.trace .p.hit{{background:#e3f5ec;color:#16694a}} .trace .p.miss{{background:#fdf3dd;color:#8a5a12}} .trace .p.tool{{background:#e6eefb;color:#1f4e9c}} .trace .a{{color:#97a3b6;margin:0 .15rem}}
.delta{{background:{NAVY};color:#fff;border-radius:10px;padding:.55rem 1rem;margin:.2rem 0 .7rem;display:flex;gap:1.6rem;flex-wrap:wrap;font-size:.85rem}}
.delta b{{font-size:1.05rem}} .hitl{{background:#fff7e6;border:1px solid #f0c36d;border-radius:10px;padding:.8rem 1rem;margin:.6rem 0}}
.mem{{background:#fff;border:1px solid #e3e8ef;border-radius:8px;padding:.4rem .6rem;margin:.25rem 0;font-size:.8rem}} .mem small{{color:{SLATE}}}
section[data-testid="stSidebar"]{{background:#fff;border-right:1px solid #e3e8ef}}
</style>""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Starting OpsGate (connecting MCP servers)...")
def _client(): return get_client()
client = _client()
TAG = re.compile(r"^\[(INTERNAL RUNBOOK|LIVE MCP / WEB SEARCH)\]\s*")
PROV = {"runbook": ("INTERNAL RUNBOOK", "b-blue"), "live_mcp": ("LIVE MCP", "b-green"), "ignorance": ("NO EVIDENCE", "b-amber"), "chat": ("CHAT", "b-grey"),
        "upload": ("ATTACHED FILE", "b-blue"), "governance": ("GOVERNED ACTION", "b-amber"), "naive": ("UNVERIFIED", "b-red")}

def badge(text, cls): return f'<span class="badge {cls}">{html.escape(str(text))}</span>'
def trace_html(steps):
    def cls(x): return "hit" if x.startswith("RAG hit") else "miss" if x.startswith(("RAG miss", "No evidence")) else "tool" if re.match(r"^[a-z_]+_[a-z_]+$", x) and not x.startswith(("Memory", "Deferred")) else ""
    return '<div class="trace">' + '<span class="a"> &rarr; </span>'.join(f'<span class="p {cls(x)}">{html.escape(x)}</span>' for x in steps) + "</div>"

# ---------- state ----------
ss = st.session_state
ss.setdefault("kpi", {"saved": 0, "turns": 0, "pct": 0.0, "lat": 0}); ss.setdefault("pending", {}); ss.setdefault("role", config.DEFAULT_ROLE if config.DEFAULT_ROLE in ROLES else "Viewer")
user_id = ss.get("uid", "demo"); store = ThreadStore(user_id)
tid = ss.get("tid")
if tid not in store.data: tid = store.list()[0]["id"] if store.list() else store.create(); ss["tid"] = tid
th = store.get(tid); doc_key = f"{user_id}:{tid}"; backend_thread = f"{user_id}-{tid}-{th['gen']}"

def slim(cmp):
    """Compact copy for the chat log (keeps what the UI renders, drops bulky evidence)."""
    c = {k: v for k, v in cmp.items() if k not in ("naive", "opsgate")}
    for b in ("naive", "opsgate"):
        x = dict(cmp[b]); t = dict(x.get("telemetry") or {}); t["contexts"] = [c[:600] for c in (t.get("contexts") or [])[:6]]; x["telemetry"] = t; c[b] = x
    return c

def safe(fn, default=None):
    try: return fn()
    except Exception as e: st.sidebar.error(f"Backend error: {e}"); return default

# ---------- sidebar ----------
with st.sidebar:
    st.markdown("### OpsGate")
    st.caption(f"Backend: {client.kind}" + (f" ({os.getenv('OPSGATE_API_URL')})" if client.kind == "api" else ""))
    if st.button("New chat", use_container_width=True, type="primary"):
        cur = store.get(ss["tid"]); ss["tid"] = ss["tid"] if cur and not cur["log"] else store.create(); st.rerun()
    shown = None
    for t in store.list():
        b = bucket(t["updated"])
        if b != shown: st.caption(b.upper()); shown = b
        c1, c2 = st.columns([8, 2], vertical_alignment="center")
        if c1.button(("● " if t["id"] == tid else "") + t["title"], key=f"sel_{t['id']}", use_container_width=True): ss["tid"] = t["id"]; st.rerun()
        if c2.button("✕", key=f"del_{t['id']}", help="Delete chat"):
            store.delete(t["id"]); ss.pop("tid", None); st.rerun()
    st.divider()
    st.markdown("**Connected MCP servers**")
    sv = safe(client.servers, {"servers": [], "tools": []})
    for s_ in sv["servers"]:
        dot = {"connected": "🟢", "failed": "🔴", "disabled": "⚪"}[s_["status"]]
        st.markdown(f"{dot} `{s_['name']}` · {s_['tools']} tools · {s_['default_risk'].replace('_', ' ').title()}")
        if s_["error"]: st.caption(f"skipped: {s_['error'][:90]}")
    with st.expander("+ Add MCP server (JSON)"):
        nm = st.text_input("Name", placeholder="my_server", key="mcp_name")
        js = st.text_area("Config", value='{\n  "command": "uvx",\n  "args": ["some-mcp-server"],\n  "default_risk": "READ_ONLY"\n}', height=130, key="mcp_json")
        if st.button("Add and hot-reload"):
            import json
            try: safe_cfg = json.loads(js); client.add_server(nm.strip(), safe_cfg); st.success("Added. Tools reloaded."); st.rerun()
            except Exception as e: st.error(str(e))
    st.divider()
    st.markdown("**Memory inspector** (shared across all threads)")
    mem = safe(lambda: client.memory(user_id), {"counts": {}, "memories": {}})
    q_mem = st.text_input("Search memories", key="memq", label_visibility="collapsed", placeholder="Search memories")
    tabs = st.tabs([f"Semantic {mem['counts'].get('semantic', 0)}", f"Episodic {mem['counts'].get('episodic', 0)}", f"Procedural {mem['counts'].get('procedural', 0)}"])
    for tab, tier in zip(tabs, ("semantic", "episodic", "procedural")):
        with tab:
            items = [m for m in mem["memories"].get(tier, []) if q_mem.lower() in m["text"].lower()]
            for m in items[:30]: st.markdown(f'<div class="mem">{html.escape(m["text"])}<br><small>{html.escape(m.get("timestamp") or "")} {("· thread " + m["source_thread"]) if m.get("source_thread") else ""}</small></div>', unsafe_allow_html=True)
            if not items: st.caption("Nothing stored yet.")
    with st.expander("Settings"):
        st.text_input("User ID", "demo", key="uid")
        if st.button("End session (extract memories)"): st.success(f"Stored {safe(lambda: client.end_session(user_id, backend_thread), {}).get('stored', 0)} memories")
        if st.button("Clear this chat"): store.clear(tid); ss["pending"].pop(tid, None); st.rerun()

# ---------- header + KPI bar ----------
k = ss["kpi"]
st.markdown('<div class="og-top"><h1>OpsGate Context Copilot</h1><span>Every message runs both pipelines. Same question, same data, side by side.</span></div>', unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 1.4, 2], vertical_alignment="bottom")
c1.markdown(f'<div class="kpi"><small>Cumulative tokens saved</small><b>{k["saved"]:,}</b></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="kpi"><small>Average savings</small><b>{(k["pct"] / k["turns"]) if k["turns"] else 0:.1f}%</b></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="kpi"><small>Latency reduction (avg)</small><b>{(k["lat"] / k["turns"]) if k["turns"] else 0:+,.0f} ms</b></div>', unsafe_allow_html=True)
c4.markdown(f'<div class="kpi"><small>Turns compared</small><b>{k["turns"]}</b></div>', unsafe_allow_html=True)
c5.selectbox("RBAC role", ROLES, key="role", help="Viewer: read-only tools. OnCall-Engineer / Admin: write tools, each needs human approval.")

st.markdown(f'<div class="card" style="margin:.7rem 0;padding:.6rem 1rem"><b style="color:{NAVY}">{html.escape(th["title"])}</b></div>', unsafe_allow_html=True)
files_here = safe(lambda: client.files(doc_key), [])
if files_here:
    with st.expander(f"Attached files ({len(files_here)})"):
        for f in files_here:
            r1, r2 = st.columns([8, 2], vertical_alignment="center"); r1.write(f"**{f['name']}** · {f['chunks']} chunks")
            if r2.button("Remove", key=f"rm_{tid}_{f['name']}"): safe(lambda: client.remove_file(doc_key, f["name"])); st.rerun()

# ---------- comparison renderer ----------
def render_comparison(cmp, pending=False):
    n, o, d = cmp["naive"], cmp["opsgate"], cmp["delta"]
    if not pending:
        st.markdown(f'<div class="delta"><span>Tokens saved<br><b>{d["tokens_saved"]:,}</b></span><span>Savings<br><b>{d["savings_pct"]}%</b></span>'
                    f'<span>Naive &rarr; OpsGate input tokens<br><b>{n["input_tokens"]:,} &rarr; {o["input_tokens"]:,}</b></span>'
                    f'<span>Latency delta<br><b>{d["latency_saved_ms"]:+,} ms</b></span><span>Role<br><b>{html.escape(cmp["user_role"])}</b></span></div>', unsafe_allow_html=True)
    L, R = st.columns(2)
    with L:
        st.markdown('<div class="card naive"><h4>Naive baseline &mdash; unoptimized</h4>' + badge(f'{n["tools_injected"]} tools injected', "b-red") + badge(f'{n["input_tokens"]:,} input tokens', "b-red")
                    + badge(f'{n["latency_ms"]:,} ms', "b-amber") + badge(f'{n["llm_calls"]} LLM calls', "b-grey") + "</div>", unsafe_allow_html=True)
        st.markdown(TAG.sub("", n["answer"]) or "_no answer_")
        with st.expander("Raw context dump"):
            t = n["telemetry"]; st.dataframe([{"element": e.replace("_", " "), "tokens": t["elements"][e]} for e in ELEMENTS], hide_index=True)
            st.code("\n\n".join(t.get("contexts", []))[:6000] or "(no retrieved context)")
    with R:
        label, cls = PROV.get(o["provenance"], ("PENDING", "b-amber")) if not pending else ("AWAITING APPROVAL", "b-amber")
        st.markdown('<div class="card ops"><h4>OpsGate &mdash; context-engineered</h4>' + badge(f'{d["savings_pct"]}% tokens saved', "b-green") + badge(f'{o["tools_loaded"]} of {o["total_tools"]} tools loaded', "b-green")
                    + badge(f'{o["input_tokens"]:,} input tokens', "b-green") + badge(f'{o["latency_ms"]:,} ms', "b-grey") + badge(label, cls) + trace_html(o["execution_trace"]) + "</div>", unsafe_allow_html=True)
        st.markdown(TAG.sub("", o["answer"]) or "_waiting for operator decision_")
        with st.expander("Execution details"):
            t = o["telemetry"]
            st.markdown("**Memory used:** " + (", ".join(f"{a} {b}" for a, b in (t.get("memory_used") or {}).items()) or "none"))
            if t.get("tool_calls"): st.dataframe([{x: c[x] for x in ("tool", "args", "ms", "out_tokens", "status")} for c in t["tool_calls"]], hide_index=True)
            if t.get("chunks"): st.dataframe(t["chunks"], hide_index=True)
            st.caption(f"KV-cache hit {t.get('cache_hit_pct', 0)}% · cleaner saved ~{t.get('cleaner_saved', 0)} tokens · route {t.get('route')}")

# ---------- history ----------
if not th["log"] and tid not in ss["pending"]:
    st.info('Try: "How do I fix Error 5042?"  ·  "What is the weather in Delhi and convert 250 USD to INR?"  ·  "Trigger rollback of payment-worker to v2.4.1"')
for m in th["log"]:
    with st.chat_message(m["role"]):
        if m.get("extra"): render_comparison(m["extra"])
        else: st.write(m["text"])
        if m.get("files"): st.caption("Attached: " + ", ".join(m["files"]))

def finish(cmp, query):
    """Completed turn: log it and update the session KPIs."""
    answer = cmp["opsgate"]["answer"]; store.append(tid, "assistant", answer, extra=slim(cmp)); d = cmp["delta"]
    ss["kpi"] = {"saved": k["saved"] + d["tokens_saved"], "turns": k["turns"] + 1, "pct": k["pct"] + d["savings_pct"], "lat": k["lat"] + d["latency_saved_ms"]}
    store.auto_title(tid, answer)

# ---------- pending approval (human in the loop) ----------
pend = ss["pending"].get(tid)
if pend:
    with st.chat_message("assistant"):
        render_comparison(pend["cmp"], pending=True)
        p = pend["cmp"]["interrupt_payload"] or {}
        st.markdown(f'<div class="hitl"><b>Approval required</b><br>{html.escape(p.get("reason", ""))}<br><code>{html.escape(str(p.get("tool", "")))}({html.escape(str(p.get("args", "")))})</code> '
                    f'&middot; risk {html.escape(str(p.get("risk", "")))} &middot; role {html.escape(str(p.get("role", "")))}</div>', unsafe_allow_html=True)
        a, r_, _ = st.columns([1.2, 1, 5])
        decision = True if a.button("Approve execution", type="primary", key=f"ap_{tid}") else False if r_.button("Reject", key=f"rj_{tid}") else None
        if decision is not None:
            with st.spinner("Resuming..."):
                res = safe(lambda: client.resume(backend_thread, decision))
            if res:
                if res["status"] == "interrupted": ss["pending"][tid] = {"cmp": res}
                else: ss["pending"].pop(tid, None); finish(res, pend["cmp"]["query"])
                st.rerun()

# ---------- input ----------
try: sub = st.chat_input("Ask about runbooks, live data, failed orders, commits, or your attached file...", accept_file="multiple", file_type=config.UPLOAD_TYPES, disabled=bool(pend))
except TypeError: st.error("This Streamlit is too old for the attach button. Run: pip install -U 'streamlit>=1.43'"); st.stop()
q, new_files = "", []
if sub: q = (getattr(sub, "text", None) or (sub.get("text") if hasattr(sub, "get") else "") or "").strip(); new_files = list(getattr(sub, "files", None) or (sub.get("files") if hasattr(sub, "get") else []) or [])
if q or new_files:
    names = []
    for f in new_files:
        try: names.append(client.upload(doc_key, f.name, f.getvalue())["name"])
        except Exception as e: st.error(str(e))
    shown_q = q or ("(attached " + ", ".join(names) + ")" if names else "")
    if shown_q:
        store.append(tid, "user", shown_q, names)
        if q:
            with st.chat_message("user"): st.write(shown_q)
            with st.spinner("Running Naive and OpsGate in parallel..."):
                cmp = safe(lambda: client.compare(q, backend_thread, user_id, ss["role"], doc_key))
            if cmp:
                if cmp["status"] == "interrupted": ss["pending"][tid] = {"cmp": cmp}
                else: finish(cmp, q)
        else:
            store.append(tid, "assistant", "Indexed " + ", ".join(names) + '. Ask me anything about it, for example: "Summarise this file".'); store.auto_title(tid)
        st.rerun()
