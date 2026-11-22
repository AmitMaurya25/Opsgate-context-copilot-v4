"""Tests for the optimisation pass (KV cache, router, cleaner fixes, sticky tools, memory update)."""
import os
os.environ.setdefault("EMBEDDING_PROVIDER", "hash"); os.environ.setdefault("SIM_THRESHOLD", "0.2"); os.environ.setdefault("GROQ_API_KEY", "")
import pytest
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from core.cleaner import clean_context
from core.router import route_by_rules
from core.cache import PrefixCache
from core.prompts import build_system

def test_cleaner_keeps_indentation_and_short_lines():
    code = "def f(x):\n    if x:\n        return 1\n    return 2\n}\n}\n"
    out = clean_context(code)
    assert "        return 1" in out and out.count("}") == 2          # indentation kept, "}" not deduped

def test_cleaner_still_dedupes_long_repeats():
    row = "txn_1001,payment-worker,5042"
    assert clean_context(f"{row}\n{row}\n{row}").count(row) == 1

@pytest.mark.parametrize("q,expected", [
    ("Hi, who are you?", "chat"), ("What is our policy on Lunar New Year holidays?", "runbook"),
    ("Who is on call for payment-worker?", "ops"), ("List the active alerts.", "ops"),
    ("Show the latest commits of the order-api service.", "ops"), ("What is the SLA policy for Sev1?", "static"),
    ("Check SLA status of payment-worker", "ops"), ("How do I fix Error 5042?", "runbook"), ("blorp", None)])
def test_router_rules(q, expected): assert route_by_rules(q) == expected

def test_prefix_cache_hits_shared_prefix_only():
    pc = PrefixCache(block=8); base = " ".join(f"w{i}" for i in range(200))
    pc.observe(base)
    _, cached = pc.observe(base + " extra question")
    assert cached >= 100                                              # shared static prefix is reused
    _, cached_broken = pc.observe("TIMESTAMP 123 " + base)           # change at the top breaks everything after it
    assert cached_broken < 16

def test_system_prompt_is_deterministic_and_ordered():
    a = build_system(["b fact", "a fact"], ["rule 2", "rule 1"], "SLA text")
    b = build_system(["a fact", "b fact"], ["rule 1", "rule 2"], "SLA text")
    assert a == b and a.index("<reference>") < a.index("RULES") < a.index("FACTS")

def test_memory_update_replaces_fact(tmp_path):
    from memory.mem0_store import LocalStore; from memory.manager import MemoryManager
    m = MemoryManager("u1", LocalStore(tmp_path / "m.json"))
    m.store.add([{"role": "user", "content": "I am vegetarian."}], "u1")
    m.update("vegetarian", "The user is vegan.")
    sem, _ = m.bootstrap()
    assert any("vegan" in x for x in sem) and not any("vegetarian" in x for x in sem)

def test_summary_turn_trigger():
    from core.summarizer import should_summarize
    msgs = [SystemMessage("s"), HumanMessage("a"), AIMessage("b"), HumanMessage("c"), AIMessage("d")]
    assert should_summarize(msgs, turn_every=2) and not should_summarize(msgs[:3], turn_every=2)

@pytest.fixture(scope="module")
def agent():
    from ingestion.contextual_indexer import build_index; build_index(reset=True)
    from core.graph import OpsGate; a = OpsGate(); yield a; a.host.close()

def test_mcp_sessions_are_persistent(agent): assert agent.host.persistent and len(agent.host.tools) >= 20

def test_static_reference_in_system_prompt_not_rag(agent):
    r = agent.ask("What is the SLA policy for Sev1?", "engineered", "s1", "s1")
    st = agent.graph.get_state({"configurable": {"thread_id": "s1"}}).values
    assert r["telemetry"]["route"] == "static" and r["telemetry"]["elements"]["rag"] == 0 and "Sev1 ack within 5 minutes" in st["messages"][0].content

def test_ops_loads_at_most_two_and_is_sticky(agent):
    r1 = agent.ask("Check the last 200 failed orders and find which service threw errors", "engineered", "o1", "o1")
    assert r1["telemetry"]["tools_in_prompt"] <= 4                    # search_tools + 3
    r2 = agent.ask("Who is on call for payment-worker?", "engineered", "o1", "o1")
    assert r2["telemetry"]["tools_in_prompt"] <= 5                    # sticky set is capped (search_tools + 4)

def test_kv_cache_grows_on_second_turn(agent):
    agent.ask("Hi, who are you?", "engineered", "k1", "k1")
    r = agent.ask("Thanks, what can you help me with?", "engineered", "k1", "k1")["telemetry"]
    assert r["cached_tokens"] > 0 and r["sent_tokens"] >= r["cached_tokens"] and r["cache_hit_pct"] > 0

def test_naive_resends_more_than_engineered(agent):
    q = "Check the last 200 failed orders and find which service threw errors"
    n = agent.ask(q, "naive", "c1", "cn")["telemetry"]; e = agent.ask(q, "engineered", "c1", "ce")["telemetry"]
    assert e["sent_tokens"] < n["sent_tokens"] and e["tools_in_prompt"] < n["tools_in_prompt"]

# ---- tracing, detail logs, output cap ----
def test_tracing_off_without_key_and_redacted_with_key(monkeypatch):
    from core.tracing import setup_tracing, trace_config
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False); assert setup_tracing()["enabled"] is False
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-not-a-real-key"); r = setup_tracing()
    assert r["enabled"] and r["redacted"] and os.environ["LANGSMITH_HIDE_INPUTS"] == "true" and os.environ["LANGSMITH_HIDE_OUTPUTS"] == "true"
    assert "alice" not in str(trace_config("engineered", "alice", "t"))        # user id is hashed
    for k in ("LANGSMITH_TRACING", "LANGSMITH_HIDE_INPUTS", "LANGSMITH_HIDE_OUTPUTS"): monkeypatch.delenv(k, raising=False)

def test_cap_tokens_truncates_with_note():
    from core.cleaner import cap_tokens
    big = "\n".join(f"row {i} payment-worker 5042" for i in range(2000))
    out, dropped = cap_tokens(big, 500)
    assert dropped > 0 and "[truncated" in out and cap_tokens("small", 500) == ("small", 0)

def test_tool_calls_and_chunks_are_logged(agent):
    t = agent.ask("Check the last 200 failed orders and find which service threw errors", "engineered", "d1", "d1")["telemetry"]
    assert t["tool_calls"] and {"tool", "ms", "out_tokens", "status"} <= set(t["tool_calls"][0]) and t["llm_log"] and t["route_reason"] == "rule"
    t2 = agent.ask("How do I fix Error 5042 in the payment worker?", "engineered", "d2", "d2")["telemetry"]
    assert t2["chunks"] and {"section", "kind", "score", "tokens"} <= set(t2["chunks"][0])

def test_sandbox_counts_come_first():
    from core.sandbox import run_programmatic_task
    class H:
        def call(self, n, a): return {"rows": [{"txn_id": i, "order_id": i, "service": "payment-worker", "error_code": 5042, "ts": "x"} for i in range(3)], "more_pages_available": False}
    assert run_programmatic_task(H(), "query_failed_orders", {})["text"].startswith("counts by service")
