"""Offline smoke tests (FakeLLM + hash embedder). Run: EMBEDDING_PROVIDER=hash SIM_THRESHOLD=0.2 pytest -q"""
import os
os.environ.setdefault("EMBEDDING_PROVIDER", "hash"); os.environ.setdefault("SIM_THRESHOLD", "0.2"); os.environ.setdefault("GROQ_API_KEY", "")
import pytest
from core.cleaner import clean_context
from ingestion.triage_splitter import chunk_markdown

def test_cleaner_keeps_facts():
    raw = "\x1b[31mERROR\x1b[0m <b>code 5042</b> at 2026-11-19 10:00:00 ======== \n\n\n\n txn  123"
    out = clean_context(raw); assert "5042" in out and "123" in out and "\x1b" not in out and "<b>" not in out and "2026-11-19" not in out

def test_triage_keeps_code_and_table():
    md = "## Fix\n\ntext here\n\n```\ndef f():\n    return 1\n```\n\n|a|b|\n|--|--|\n|1|2|\n"
    kinds = [c["kind"] for c in chunk_markdown(md)]; assert "code" in kinds and "table" in kinds
    assert any("    return 1" in c["text"] for c in chunk_markdown(md))

def test_sandbox_shrinks_output():
    from core.mcp_host import MCPHost; from core.sandbox import run_programmatic_task
    out = run_programmatic_task(MCPHost().load(), "db_logs_query_failed_orders", {"limit": 200})
    assert out["rows"] == 200 and len(out["text"]) < out["raw_chars"] / 5

@pytest.fixture(scope="module")
def agent():
    from ingestion.contextual_indexer import build_index; build_index(reset=True)
    from core.graph import OpsGate; return OpsGate()

def test_chat_uses_no_tools(agent):
    r = agent.ask("Hi, who are you?", "engineered", "t1"); assert r["telemetry"]["route"] == "chat" and r["telemetry"]["tools_in_prompt"] == 0

def test_out_of_scope_refuses_when_rag_and_mcp_have_nothing(agent):
    """Route C: RAG misses AND no tool produces evidence -> the Ignorance Clause (never a made-up answer)."""
    from langchain_core.messages import AIMessage
    class NoTools:
        def bind_tools(self, tools): return self
        def invoke(self, *a, **k): return AIMessage(content="I have no idea.")
    real, agent.llm = agent.llm, NoTools()
    try: r = agent.ask("What is our policy on Lunar New Year holidays?", "engineered", "t2")
    finally: agent.llm = real
    assert "cannot verify" in r["answer"].lower() and r["provenance"] == "ignorance"

def test_ops_defers_tools(agent):
    r = agent.ask("Check the last 200 failed orders and find which service threw errors", "engineered", "t3")
    assert r["telemetry"]["tools_in_prompt"] <= 4 and r["telemetry"]["total_tools"] == len(agent.host.tools) >= 20
