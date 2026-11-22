"""LLM + embedding factories. Falls back to offline fakes when no GROQ_API_KEY (used by tests)."""
import hashlib, re
import numpy as np
from langchain_core.messages import AIMessage
from core import config

IGNORANCE = "I cannot verify this from internal runbooks or live MCP tools."
UPLOAD_IGNORANCE = "I could not find that in the attached file(s)."

def have_llm() -> bool:
    return bool(config.GROQ_API_KEY)

def get_llm(model: str | None = None, temperature: float = 0.0):
    if have_llm():
        from langchain_groq import ChatGroq
        return ChatGroq(model=model or config.GROQ_MODEL, temperature=temperature, api_key=config.GROQ_API_KEY)
    return FakeLLM()

def _schema_props(t):
    sch = t.get("parameters") if isinstance(t, dict) else (t.args_schema if isinstance(getattr(t, "args_schema", None), dict) else t.args_schema.model_json_schema())
    sch = sch or {}
    return sch.get("properties", {}), sch.get("required", [])

def _fake_args(tool, text: str) -> dict:
    """Offline stand-in for the model filling tool arguments from the question (regex heuristics, tests/demo only)."""
    props, req = _schema_props(tool); args = {}
    codes = re.findall(r"\b[A-Z]{3}\b", text); nums = re.findall(r"\d+(?:\.\d+)?", text)
    for p in req:
        pl, typ = p.lower(), (props.get(p, {}).get("type") or "string")
        if pl == "city": m = re.search(r"\b(?:in|for|at) ([A-Z][a-zA-Z]+)", text); args[p] = m.group(1) if m else "Delhi"
        elif pl == "amount": args[p] = float(nums[0]) if nums else 1.0
        elif pl == "from_currency": args[p] = codes[0] if codes else "USD"
        elif pl == "to_currency": args[p] = codes[1] if len(codes) > 1 else "INR"
        elif pl == "timezone": m = re.search(r"\b[A-Z][a-z]+/[A-Z][a-z_]+\b", text); args[p] = m.group(0) if m else "UTC"
        elif pl in ("query", "q"): args[p] = text[:200]
        elif pl == "service": m = re.search(r"\b(payment-worker|order-api|inventory-sync)\b", text); args[p] = m.group(1) if m else "payment-worker"
        elif pl == "version": args[p] = "v2.4.1"
        elif pl == "summary": args[p] = text[:80]
        elif pl == "path": args[p] = "main.py"
        elif typ == "integer": args[p] = int(nums[0]) if nums else 1
        elif typ == "number": args[p] = float(nums[0]) if nums else 1.0
        else: args[p] = "x"
    return args

_FSTOP = set("the a an of to in on for and or is are was were be do does did how what which who when where why can could would should i me my we our you your it its this that with from by at as about into please get show".split())
def _words(t): return {w for w in re.findall(r"[a-z0-9]+", t.lower().replace("_", " ")) if w not in _FSTOP}

class FakeLLM:
    """Deterministic offline stand-in so the pipeline can be tested without an API key."""
    def __init__(self, tools=None): self._tools = tools or []
    def bind_tools(self, tools): return FakeLLM(tools)
    @staticmethod
    def _name(t): return t["name"] if isinstance(t, dict) else t.name
    def invoke(self, messages, *a, **k):
        msgs = messages if isinstance(messages, list) else [messages]
        last_human = next((i for i in range(len(msgs) - 1, -1, -1) if getattr(msgs[i], "type", "") == "human"), -1)
        full = msgs[last_human].content if last_human >= 0 else ""
        q = re.search(r"<question>\n(.*?)\n</question>", full, re.S); last_user = q.group(1) if q else full
        last_tool = [m for m in msgs[last_human + 1:] if getattr(m, "type", "") == "tool"]
        names = [self._name(t) for t in self._tools]
        if self._tools:
            if not last_tool and "search_tools" in names and len(names) == 1:
                return AIMessage(content="", tool_calls=[{"name": "search_tools", "args": {"query": last_user}, "id": "c1"}])
            if not last_tool or last_tool[-1].name == "search_tools":
                cands = [t for t in self._tools if self._name(t) != "search_tools"]
                if cands:
                    qw = _words(last_user)
                    ql = " ".join(re.findall(r"[a-z0-9]+", last_user.lower()))
                    def score(t):
                        d = t["description"] if isinstance(t, dict) else (t.description or ""); toks = self._name(t).split("_")
                        boost = 10 if any(" ".join(toks[k:]) in ql for k in range(1, len(toks) - 1)) else 0      # the question names the tool ("trigger rollback")
                        return boost + len(qw & _words(self._name(t) + " " + d))
                    ranked = sorted(cands, key=lambda t: -score(t)); best = score(ranked[0])
                    pick = [t for t in ranked if score(t) >= max(1, 0.3 * best)][:3] if best > 0 else ranked[:1]
                    return AIMessage(content="", tool_calls=[{"name": self._name(t), "args": _fake_args(t, last_user), "id": f"c{i}"} for i, t in enumerate(pick, 2)])
        if last_tool: return AIMessage(content="Result: " + " | ".join(m.content[:220] for m in last_tool[:3]))
        text = " ".join(getattr(m, "content", "") for m in msgs)
        if "Summarize" in text or "summary" in text.lower():
            return AIMessage(content="Decisions: none. Facts: see history. Preferences: none. Pending: none. Goal: ongoing.")
        doc = re.search(r"<document [^>]*>\n(.*?)\n</document>", full, re.S)
        if doc: return AIMessage(content="(offline) " + " ".join(doc.group(1).split())[:260])
        return AIMessage(content=f"(offline) {last_user[:200]}")

_HASH_STOP = set("the a an of to in on for and or is are was were be do does did how what which who when where why can could would should i me my we our you your it its this that with from by at as about into".split())

class HashEmbedder:
    """Offline bag-of-words hashing embedder (stopwords dropped so function words do not create fake similarity). Weak quality; for tests only."""
    dim = 256
    def embed(self, texts):
        out = []
        for t in texts:
            v = np.zeros(self.dim)
            for w in (w for w in re.findall(r"[a-z0-9_]+", t.lower()) if w not in _HASH_STOP):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1
            n = np.linalg.norm(v); out.append((v / n if n else v).tolist())
        return out

class LocalEmbedder:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(config.EMBEDDING_MODEL)
    def embed(self, texts):
        return self.m.encode(texts, normalize_embeddings=True).tolist()

_emb = None
def get_embedder():
    global _emb
    if _emb is None:
        _emb = HashEmbedder() if config.EMBEDDING_PROVIDER == "hash" else LocalEmbedder()
    return _emb

def cosine(a, b):
    a, b = np.asarray(a), np.asarray(b)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d else 0.0
