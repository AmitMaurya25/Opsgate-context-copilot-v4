"""One client interface for the UI: HttpClient talks to the FastAPI backend (OPSGATE_API_URL set), LocalClient runs the same code in-process."""
import base64, os

class LocalClient:
    kind = "in-process"
    def __init__(self): self._a = None
    @property
    def agent(self):
        if self._a is None:
            from core.graph import OpsGate; self._a = OpsGate()
        return self._a
    def health(self): return {"status": "ok", "tools": len(self.agent.host.tools), "servers": len(self.agent.host.registry)}
    def compare(self, query, thread_id, user_id, role, doc_key=""): return self.agent.run_comparison_turn(query, thread_id, user_id, role, doc_key)
    def resume(self, thread_id, approved): return self.agent.resume_comparison(thread_id, approved)
    def memory(self, user_id):
        inv = self.agent.mem(user_id).inventory(); return {"counts": {k: len(v) for k, v in inv.items()}, "memories": inv}
    def servers(self):
        a = self.agent; return {"servers": a.host.servers(), "tools": [{"name": n, "server": a.host.server_of[n], "risk": a.host.risk[n]} for n in a.host.tools]}
    def add_server(self, name, cfg): self.agent.add_mcp_server(name, cfg); return self.servers()
    def upload(self, doc_key, name, data):
        from core import uploads; return uploads.save_and_index(doc_key, name, data)
    def files(self, doc_key):
        from core import uploads; return uploads.list_files(doc_key)
    def remove_file(self, doc_key, name):
        from core import uploads; uploads.remove_file(doc_key, name)
    def end_session(self, user_id, thread_id): return {"stored": len(self.agent.end_session(user_id, "engineered", f"{thread_id}-engineered"))}

class HttpClient:
    kind = "api"
    def __init__(self, base): self.base = base.rstrip("/")
    def _r(self, method, path, **kw):
        import httpx
        r = httpx.request(method, self.base + path, timeout=kw.pop("timeout", 180), **kw)
        if r.status_code >= 400:
            try: detail = r.json().get("detail", r.text)
            except Exception: detail = r.text
            raise RuntimeError(f"{r.status_code}: {detail}")
        return r.json()
    def health(self): return self._r("GET", "/v1/health", timeout=10)
    def compare(self, query, thread_id, user_id, role, doc_key=""): return self._r("POST", "/v1/chat/compare", json={"query": query, "thread_id": thread_id, "user_id": user_id, "user_role": role, "doc_key": doc_key})
    def resume(self, thread_id, approved): return self._r("POST", "/v1/chat/resume", json={"thread_id": thread_id, "approved": approved})
    def memory(self, user_id): return self._r("GET", f"/v1/memory/{user_id}")
    def servers(self): return self._r("GET", "/v1/mcp/servers")
    def add_server(self, name, cfg): return self._r("POST", "/v1/mcp/servers", json={"name": name, "config": cfg}, timeout=240)
    def upload(self, doc_key, name, data): return self._r("POST", "/v1/upload", json={"doc_key": doc_key, "name": name, "content_b64": base64.b64encode(data).decode()})
    def files(self, doc_key): return self._r("GET", "/v1/files", params={"doc_key": doc_key})["files"]
    def remove_file(self, doc_key, name): self._r("POST", "/v1/files/remove", json={"doc_key": doc_key, "name": name})
    def end_session(self, user_id, thread_id): return self._r("POST", "/v1/session/end", json={"user_id": user_id, "thread_id": thread_id})

def get_client():
    url = os.getenv("OPSGATE_API_URL", "").strip()
    return HttpClient(url) if url else LocalClient()
