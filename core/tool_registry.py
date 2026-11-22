"""Deferred loading: the host keeps all schemas; the LLM first sees only `search_tools`."""
import json, re
from functools import lru_cache
from langchain_core.tools import tool
from core import config
from core.llm import get_embedder, cosine
from core.telemetry import count_tokens

STOP = set("the a an of to in on for and or is are was what which who find check show me my our last get give do how".split())
def _words(t): return {w for w in re.findall(r"[a-z0-9]+", t.lower().replace("_", " ")) if w not in STOP}

def schema_of(t) -> dict:
    a = t.args_schema if isinstance(t.args_schema, dict) else t.args_schema.model_json_schema()
    return {"name": t.name, "description": t.description, "parameters": a}

class ToolRegistry:
    def __init__(self, tools: dict, risk: dict | None = None, server_of: dict | None = None):
        self.tools = tools
        self.risks, self.servers = dict(risk or {}), dict(server_of or {})
        self.emb = get_embedder()
        self.entries = {n: {"text": f"{n} {t.description}", "words": _words(f"{n} {t.description}"), "schema": schema_of(t)} for n, t in tools.items()}
        vecs = self.emb.embed([e["text"] for e in self.entries.values()])
        for e, v in zip(self.entries.values(), vecs): e["vec"] = v
        self.order = {n: i for i, n in enumerate(self.entries)}       # fixed registry order -> stable schema order -> stable KV prefix
        self._schema_tok: dict = {}
    def risk(self, name: str) -> str: return self.risks.get(name, "PRIVILEGED_WRITE")       # unknown tool = treat as privileged
    def server(self, name: str) -> str: return self.servers.get(name, "")
    def allowed(self, role: str) -> set:
        """Tool names this RBAC role may even see (a Viewer never has write tools loaded)."""
        from core.guardrails import role_allows
        return {n for n in self.entries if role_allows(role, self.risk(n))}
    def ordered(self, names) -> list[str]:
        """Always bind tools in registry order, never in score order (same tool set => identical prompt prefix)."""
        return sorted(set(names), key=lambda n: self.order.get(n, 10**6))
    @lru_cache(maxsize=256)
    def _qvec(self, query: str): return tuple(self.emb.embed([query])[0])
    def search(self, query: str, k: int = 2, allowed: set | None = None) -> list[str]:
        qw, qv = _words(query), self._qvec(query)
        scored = []
        for n, e in self.entries.items():
            if allowed is not None and n not in allowed: continue
            kw = len(qw & e["words"]) / max(1, len(qw))
            scored.append((0.6 * kw + 0.4 * cosine(qv, e["vec"]), n))
        scored.sort(reverse=True)
        return [n for _, n in scored[:k]]
    def search_multi(self, query: str, total: int = 3, allowed: set | None = None) -> list[str]:
        """Multi-intent queries ("weather in Delhi AND convert USD to INR"): best tool per clause first, then best overall."""
        out: list[str] = []
        for clause in [c for c in re.split(r"\band\b|[,;?]|\bthen\b", query, flags=re.I) if len(_words(c)) >= 1]:
            for n in self.search(clause, 1, allowed):
                if n not in out: out.append(n)
        for n in self.search(query, total, allowed):
            if n not in out: out.append(n)
        return out[:total]
    def schema_tokens(self, names) -> int:
        for n in names:
            if n not in self._schema_tok: self._schema_tok[n] = count_tokens(json.dumps(self.entries[n]["schema"]))
        return sum(self._schema_tok[n] for n in names)
    def make_search_tool(self):
        reg = self
        @tool
        def search_tools(query: str) -> str:
            """Find which available tools can handle a request. Returns tool names; their schemas are then loaded."""
            return json.dumps(reg.search(query, k=2))
        return search_tools
    def use_deferred(self) -> bool:
        return len(self.tools) >= config.DEFER_MIN_TOOLS
