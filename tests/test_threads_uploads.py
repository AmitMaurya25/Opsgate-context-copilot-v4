"""Threads (pure logic) + attached-file retrieval and routing. Run: EMBEDDING_PROVIDER=hash SIM_THRESHOLD=0.2 pytest -q tests"""
import os
os.environ.setdefault("EMBEDDING_PROVIDER", "hash"); os.environ.setdefault("SIM_THRESHOLD", "0.2"); os.environ.setdefault("GROQ_API_KEY", "")
import pytest
from core.threads import ThreadStore, heuristic_title, unique_title, bucket

DOC = ("# Refund Policy\n\nRefunds over 500 dollars need manager approval. Invoice INV-2041 is dated 2026-11-03.\n\n"
       "# Escalation\n\nEscalate sev1 incidents to the on-call manager within 15 minutes.\n")

def test_heuristic_title_uses_content():
    assert heuristic_title("How do I fix Error 5042 in the payment worker?") == "Fix Error 5042 Payment Worker"
    assert heuristic_title("summarise this", ["Q3_incident_report.pdf"]) == "Q3 Incident Report"
    assert heuristic_title("Hi, who are you?") == "New chat"

def test_titles_are_unique_and_persist(tmp_path):
    st = ThreadStore("u", tmp_path); ids = []
    for _ in range(2):
        t = st.create(); st.append(t, "user", "How do I fix Error 5042?"); st.auto_title(t); ids.append(t)
    a, b = st.get(ids[0])["title"], st.get(ids[1])["title"]
    assert a != b and b.startswith(a)
    assert [t["title"] for t in ThreadStore("u", tmp_path).list()] and ThreadStore("u", tmp_path).get(ids[0])["log"]   # survives a reload

def test_manual_rename_is_never_overwritten(tmp_path):
    st = ThreadStore("u", tmp_path); t = st.create(); st.rename(t, "My name")
    for _ in range(3): st.append(t, "user", "check failed orders")
    st.auto_title(t); assert st.get(t)["title"] == "My name"

def test_corrupt_store_does_not_crash(tmp_path):
    (tmp_path / "u.json").write_text("{nope"); assert ThreadStore("u", tmp_path).list() == []

def test_bucket(): assert bucket(1_000_000, now=1_000_000 + 3 * 86400) == "Previous 7 days"

@pytest.fixture(scope="module")
def agent():
    from ingestion.contextual_indexer import build_index; build_index(reset=True)
    from core.graph import OpsGate; return OpsGate()

@pytest.fixture()
def keyed():
    from core import uploads
    k = "pytest:thread1"; uploads.save_and_index(k, "policy.txt", DOC.encode()); yield k; uploads.delete_thread_files(k)

def test_upload_is_isolated_per_thread(keyed):
    from core import uploads
    assert uploads.has_files(keyed) and not uploads.has_files("pytest:other")
    assert uploads.retrieve_uploads("escalate sev1 manager", "pytest:other", threshold=0.0) == []
    assert uploads.retrieve_uploads("escalate sev1 manager", keyed)[0]["source"] == "policy.txt"

def test_upload_keeps_dates_exact(keyed, agent):
    r = agent.ask("What date is invoice INV-2041?", "engineered", "pt", "pytest-dates", keyed)
    assert r["telemetry"]["route"] == "doc" and any(c["source"] == "policy.txt" for c in r["telemetry"]["chunks"])
    assert "2026-11-03" in "".join(d["text"] for d in __import__("core.uploads", fromlist=["x"]).retrieve_uploads("INV-2041 dated", keyed))

def test_doc_route_only_when_files_attached(keyed, agent):
    assert agent.ask("What is the escalation window for sev1?", "engineered", "pt", "pytest-a", keyed)["telemetry"]["route"] == "doc"
    assert agent.ask("Hi, who are you?", "engineered", "pt", "pytest-b", keyed)["telemetry"]["route"] == "chat"        # greeting rule still wins
    assert agent.ask("Show the latest commits of order-api", "engineered", "pt", "pytest-c", keyed)["telemetry"]["route"] == "ops"   # ops rule still wins
    assert agent.ask("What is the escalation window for sev1?", "engineered", "pt", "pytest-d", "")["telemetry"]["route"] != "doc"  # no files -> old behaviour
