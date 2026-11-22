"""FastAPI backend. The Streamlit UI (or any client) talks to this; the agent, MCP host, memory and vector stores live here only.
Run: uvicorn api.server:app --port 8000"""
import base64, sys, threading
from contextlib import asynccontextmanager
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from core import config, uploads, guardrails

_state: dict = {"agent": None}
_lock = threading.Lock()

def get_agent():
    with _lock:
        if _state["agent"] is None:
            from core.graph import OpsGate
            _state["agent"] = OpsGate()
        return _state["agent"]

@asynccontextmanager
async def lifespan(app):
    yield
    if _state["agent"] is not None: _state["agent"].host.close()

app = FastAPI(title="OpsGate Context Copilot API", version="4.0", lifespan=lifespan)

class CompareReq(BaseModel):
    query: str = Field(min_length=1)
    thread_id: str
    user_id: str = "demo"
    user_role: str = config.DEFAULT_ROLE
    doc_key: str = ""
class ResumeReq(BaseModel): thread_id: str; approved: bool
class ServerReq(BaseModel): name: str; config: dict
class UploadReq(BaseModel): doc_key: str; name: str; content_b64: str
class FileReq(BaseModel): doc_key: str; name: str
class EndReq(BaseModel): user_id: str = "demo"; thread_id: str

@app.get("/v1/health")
def health():
    a = get_agent(); return {"status": "ok", "tools": len(a.host.tools), "servers": len(a.host.registry), "llm": "groq" if config.GROQ_API_KEY else "offline-fake", "roles": guardrails.ROLES}

@app.post("/v1/chat/compare")
def chat_compare(req: CompareReq):
    """Runs Naive and OpsGate simultaneously. status="interrupted" + interrupt_payload when a PRIVILEGED_WRITE tool needs approval."""
    return get_agent().run_comparison_turn(req.query, req.thread_id, req.user_id, req.user_role, req.doc_key)

@app.post("/v1/chat/resume")
def chat_resume(req: ResumeReq):
    try: return get_agent().resume_comparison(req.thread_id, req.approved)
    except KeyError as e: raise HTTPException(404, str(e))

@app.get("/v1/memory/{user_id}")
def memory(user_id: str):
    inv = get_agent().mem(user_id).inventory()
    return {"user_id": user_id, "counts": {k: len(v) for k, v in inv.items()}, "memories": inv}

@app.get("/v1/mcp/servers")
def mcp_servers():
    a = get_agent(); return {"servers": a.host.servers(), "tools": [{"name": n, "server": a.host.server_of[n], "risk": a.host.risk[n]} for n in a.host.tools]}

@app.post("/v1/mcp/servers")
def mcp_add(req: ServerReq):
    """Hot-adds a server to mcp_servers.json, restarts sessions and rebuilds the tool registry. No code change needed."""
    try: entry = get_agent().add_mcp_server(req.name, req.config)
    except ValueError as e: raise HTTPException(422, str(e))
    return {"added": req.name, "entry": entry, **mcp_servers()}

@app.post("/v1/upload")
def upload(req: UploadReq):
    try: return uploads.save_and_index(req.doc_key, req.name, base64.b64decode(req.content_b64))
    except Exception as e: raise HTTPException(422, str(e))
@app.get("/v1/files")
def files(doc_key: str): return {"files": uploads.list_files(doc_key)}
@app.post("/v1/files/remove")
def files_remove(req: FileReq): uploads.remove_file(req.doc_key, req.name); return {"removed": req.name}
@app.post("/v1/session/end")
def session_end(req: EndReq):
    a = get_agent(); return {"stored": len(a.end_session(req.user_id, "engineered", f"{req.thread_id}-engineered"))}
