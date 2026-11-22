"""Memory backends with 3 tiers shared across ALL threads. Mem0Store = real Mem0 (needs GROQ_API_KEY); LocalStore = offline JSON store (tests/demo).
Scopes: Semantic + Episodic -> user_id; Procedural -> agent_id="opsgate-copilot" (shared operating rules). Nothing is scoped by thread/run id;
episodes only RECORD their source thread in metadata."""
import json, re, threading, time, uuid
from datetime import datetime, timezone
from core import config
from core.llm import get_embedder, cosine, have_llm

CUSTOM_INSTRUCTIONS = """Extract durable facts about the USER only. Ignore chit-chat and general-knowledge questions.
Rewrite first person as third person ("The user ..."). Start every memory with exactly one or two tags:
[Semantic] stable facts (identity, location, diet, allergies, role, goals)
[Procedural] interaction rules/preferences (language, tone, step-by-step, format)
[Episodic] events and decisions (what happened, what was tried, what was built)
A fact may carry two tags. Example: "[Semantic][Episodic] The user moved from Delhi to Bore." """

TAG_RE = re.compile(r"\[(Semantic|Episodic|Procedural)\]", re.I)
def tags_of(text): return [t.capitalize() for t in TAG_RE.findall(text)]
def strip_tags(text): return TAG_RE.sub("", text).strip()

TIERS = ("semantic", "episodic", "procedural")
def utc_now() -> str: return datetime.now(timezone.utc).isoformat(timespec="seconds")
def tier_of(rec: dict) -> str:
    t = (rec.get("metadata") or {}).get("memory_type")
    if t in TIERS: return t
    tags = tags_of(rec.get("memory", "")); return tags[0].lower() if tags else "semantic"
def _meta(tier, thread_id=None):
    m = {"memory_type": tier}
    if tier == "episodic": m.update(source_thread=thread_id or "", timestamp=utc_now())
    return m

class LocalStore:
    """Offline store. Rule-based tagger stands in for Mem0's LLM extraction."""
    def __init__(self, path=None):
        self.path = path or (config.MEM0_DIR / "local_memories.json"); self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        try: self.items = json.loads(self.path.read_text()) if self.path.exists() else []
        except Exception: self.items = []
    def _save(self): self.path.write_text(json.dumps(self.items, indent=1))
    @staticmethod
    def _tag(line):
        l = line.lower()
        if re.search(r"\b(prefer|always|please use|step by step|step-by-step|don't use|do not use)\b", l): return "[Procedural]"
        if re.search(r"\b(i am|i'm|my name|i live|i work|allergic|vegetarian|vegan|i moved)\b", l): return "[Semantic]"
        if re.search(r"\b(i built|we discussed|i tried|i decided|yesterday|last week|project)\b", l): return "[Episodic]"
        return None
    def _put(self, memory, tier, user_id=None, agent_id=None, thread_id=None):
        rec = {"id": uuid.uuid4().hex[:8], "user_id": user_id, "agent_id": agent_id, "memory": memory, "ts": time.time(), "metadata": _meta(tier, thread_id)}
        with self._lock: self.items.append(rec); self._save()
        return rec
    def add_memory(self, text, tier, user_id, thread_id=None):
        """Tier-aware write. Procedural is stored under the shared agent_id, everything else under the user."""
        if tier == "procedural": return self._put(text.strip(), tier, agent_id=config.AGENT_ID)
        return self._put(text.strip(), tier, user_id=user_id, thread_id=thread_id)
    def add(self, messages, user_id):
        out = []
        for m in messages:
            role, text = (m["role"], m["content"]) if isinstance(m, dict) else (m.type, m.content)
            if role not in ("user", "human"): continue
            for sent in re.split(r"(?<=[.!?])\s+", text):
                tag = self._tag(sent)
                if tag:
                    tier = tag.strip("[]").lower()
                    out.append(self._put(f"{tag} The user said: {sent.strip()}", tier, user_id=None if tier == "procedural" else user_id, agent_id=config.AGENT_ID if tier == "procedural" else None))
        return out
    def add_raw(self, text, user_id, tag="[Semantic]"):
        """Store an already-tagged fact verbatim (used by update(), so a replacement is never lost to the tagger)."""
        tier = tag.strip("[]").lower()
        return [self._put(f"{tag} {text.strip()}", tier, user_id=None if tier == "procedural" else user_id, agent_id=config.AGENT_ID if tier == "procedural" else None)]
    def get_all(self, user_id):
        with self._lock: return [i for i in self.items if i.get("user_id") == user_id or i.get("agent_id") == config.AGENT_ID]
    def search(self, query, user_id, limit=5, tier=None):
        mine = [m for m in self.get_all(user_id) if tier is None or tier_of(m) == tier]
        if not mine: return []
        emb = get_embedder(); q = emb.embed([query])[0]; vecs = emb.embed([strip_tags(m["memory"]) for m in mine])
        res = [{**m, "score": round(cosine(q, v), 4)} for m, v in zip(mine, vecs)]
        return sorted(res, key=lambda r: -r["score"])[:limit]
    def delete(self, mid):
        with self._lock: self.items = [i for i in self.items if i["id"] != mid]; self._save()

class Mem0Store:
    """Real Mem0 (Groq LLM, local HF embedder, Chroma or Qdrant vectors). UNVERIFIED without an API key: check against the installed mem0ai version."""
    def __init__(self):
        from mem0 import Memory
        vs = ({"provider": "qdrant", "config": {"collection_name": "opsgate_mem", "host": config.QDRANT_HOST, "port": config.QDRANT_PORT}} if config.MEM0_VECTOR_STORE == "qdrant"
              else {"provider": "chroma", "config": {"collection_name": "opsgate_mem", "path": str(config.MEM0_DIR / "mem0_chroma")}})
        cfg = {"llm": {"provider": "groq", "config": {"model": config.GROQ_MODEL, "api_key": config.GROQ_API_KEY}},
               "embedder": {"provider": "huggingface", "config": {"model": config.EMBEDDING_MODEL}},
               "vector_store": vs, "custom_instructions": CUSTOM_INSTRUCTIONS}
        self.m = Memory.from_config(cfg)
    @staticmethod
    def _res(r): return r.get("results", r) if isinstance(r, dict) else r
    def add_memory(self, text, tier, user_id, thread_id=None):
        """Tier writes are verbatim (infer=False): the tier is decided by OpsGate's classifier, not re-guessed by Mem0."""
        md = _meta(tier, thread_id)
        if tier == "procedural": r = self.m.add([{"role": "user", "content": text}], agent_id=config.AGENT_ID, infer=False, metadata=md)
        else: r = self.m.add([{"role": "user", "content": text}], user_id=user_id, infer=False, metadata=md)
        return self._res(r)
    def add(self, messages, user_id):
        msgs = [m if isinstance(m, dict) else {"role": "user" if m.type == "human" else "assistant", "content": m.content} for m in messages]
        return self._res(self.m.add(msgs, user_id=user_id))
    def add_raw(self, text, user_id, tag="[Semantic]"):
        tier = tag.strip("[]").lower()
        if tier == "procedural": return self.add_memory(text, tier, user_id)
        try: r = self.m.add([{"role": "user", "content": f"{tag} {text}"}], user_id=user_id, infer=False, metadata=_meta(tier))
        except TypeError: r = self.m.add([{"role": "user", "content": f"{tag} {text}"}], user_id=user_id)
        return self._res(r)
    def _scoped(self, fn_new, fn_old, **scope):
        try: return self._res(fn_new(scope))
        except TypeError: return self._res(fn_old(scope))
    def get_all(self, user_id):
        out = []
        for scope in ({"user_id": user_id}, {"agent_id": config.AGENT_ID}):
            out += self._scoped(lambda s: self.m.get_all(filters=s), lambda s: self.m.get_all(**s), **scope)
        return out
    def search(self, query, user_id, limit=5, tier=None):
        out = []
        for scope in ({"user_id": user_id}, {"agent_id": config.AGENT_ID}):
            out += self._scoped(lambda s: self.m.search(query, filters=s, limit=limit), lambda s: self.m.search(query, limit=limit, **s), **scope)
        out = [r for r in out if tier is None or tier_of(r) == tier]
        return sorted(out, key=lambda r: -r.get("score", 0))[:limit]
    def delete(self, mid): self.m.delete(mid)

def get_store():
    return Mem0Store() if have_llm() else LocalStore()
