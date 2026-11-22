"""Memory lifecycle across ALL threads: recall (every turn, 3 tiers) -> inject; classify -> dedupe -> write (async, after the response).
Semantic = durable facts (user_id). Episodic = timestamped events (user_id, source thread recorded in metadata). Procedural = rules (agent_id, shared)."""
import hashlib, json, re, threading
from core import config
from memory.mem0_store import get_store, tier_of, tags_of, strip_tags, TIERS
from memory.dedup import dedupe, is_duplicate

PENDING: list[threading.Thread] = []
def wait_for_writes(timeout: float = 15.0):
    for t in list(PENDING): t.join(timeout)
    PENDING[:] = [t for t in PENDING if t.is_alive()]

PROCEDURAL = re.compile(r"\b(always|never|whenever|from now on|every time|each time|make sure (?:to|you)|remember to)\b", re.I)
SEMANTIC = re.compile(r"\b(my default|my (?:name|team|role|timezone|company)|i prefer|i am an?\b|i'm an?\b|i work|i live|we use|we run|(?:our|the) (?:prod\w*|database|stack|cluster)\b.*\b(?:runs?|uses?|is on)\b|runs on)\b", re.I)
EPISODE = re.compile(r"\b(was traced|root cause|resolved|we fixed|fixed it|restarted|rolled back|we decided|postmortem)\b", re.I)
QUESTION = re.compile(r"\?\s*$|^\s*(what|how|who|when|where|why|which|can|could|show|check|list|find|get|tell)\b", re.I)

PROV_TAG = re.compile(r"^\[[^\]]+\]\s*")
def _sentences(text): return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.strip()) > 8]

def extract(question: str, answer: str = "", provenance: str = "", today: str = "") -> list[tuple[str, str]]:
    """Rule-based classifier used offline (and as the LLM fallback). Returns [(tier, text)]."""
    out = []
    for s in _sentences(question):
        if PROCEDURAL.search(s) and not (QUESTION.search(s) and not re.search(r"\b(always|never|from now on)\b", s, re.I)): out.append(("procedural", s.rstrip(".")))
        elif SEMANTIC.search(s): out.append(("semantic", "User states: " + s.rstrip(".")))
        elif EPISODE.search(s): out.append(("episodic", f"On {today}: " + s.rstrip(".")))
    if not out and provenance in ("runbook", "live_mcp") and answer:
        out.append(("episodic", f"On {today}: asked '{question.strip()[:110]}' and got: {PROV_TAG.sub('', answer).strip()[:150]}"))
    return out

def extract_llm(question: str, answer: str, provenance: str, today: str):
    from core.llm import get_llm
    prompt = ("Extract durable memories from this exchange as JSON: a list of {\"type\": \"semantic|episodic|procedural\", \"text\": str}.\n"
              "semantic = stable user facts or preferences; episodic = a timestamped event or outcome (start with 'On " + today + ":'); procedural = a reusable rule or workflow the user wants applied.\n"
              "Return [] when nothing is worth keeping. Output JSON only.\n"
              f"<user>{question[:800]}</user>\n<assistant>{answer[:600]}</assistant>")
    raw = get_llm(config.SUMMARIZER_MODEL).invoke(prompt).content
    data = json.loads(raw[raw.index("["): raw.rindex("]") + 1])
    return [(d["type"], d["text"]) for d in data if d.get("type") in TIERS and d.get("text")]

class MemoryManager:
    def __init__(self, user_id: str, store=None):
        self.user_id, self.store = user_id, store or get_store()
        self._dedupe_cache: dict = {}
    # ---- read ----
    def _by_tier(self, tier): return [r for r in self.store.get_all(self.user_id) if tier_of(r) == tier]
    def _dedupe_once(self, records):
        """Same memories => same merged text, with no repeat LLM call."""
        key = hashlib.md5(json.dumps([(r.get("id"), r["memory"]) for r in records], sort_keys=True).encode()).hexdigest()
        if key not in self._dedupe_cache: self._dedupe_cache[key] = sorted(dedupe(records))
        return self._dedupe_cache[key]
    def bootstrap(self) -> tuple[list[str], list[str]]:
        return self._dedupe_once(self._by_tier("semantic")), self._dedupe_once(self._by_tier("procedural"))
    def recall(self, query: str, k: int = 3, threshold: float | None = None) -> dict:
        """Targeted search per tier, EVERY turn, regardless of thread."""
        th = config.MEMORY_THRESHOLD if threshold is None else threshold
        out = {}
        for tier in TIERS:
            hits = [r for r in self.store.search(query, self.user_id, limit=k * 3, tier=tier) if r.get("score", 0) >= th]
            out[tier] = [strip_tags(r["memory"]) for r in hits[:k]]
        return out
    def episodic(self, query: str, k: int = 2, threshold: float | None = None) -> list[str]:
        return self.recall(query, k, threshold)["episodic"]
    def inventory(self) -> dict:
        """Everything stored, grouped by tier, for the UI inspector."""
        out = {t: [] for t in TIERS}
        for r in sorted(self.store.get_all(self.user_id), key=lambda r: -r.get("ts", 0)):
            md = r.get("metadata") or {}
            out[tier_of(r)].append({"id": r.get("id"), "text": strip_tags(r["memory"]), "timestamp": md.get("timestamp", ""), "source_thread": md.get("source_thread", "")})
        return out
    # ---- write ----
    def write_turn(self, question: str, answer: str = "", thread_id: str = "", provenance: str = "") -> list[dict]:
        from datetime import date
        today = date.today().isoformat(); items = []
        try:
            from core.llm import have_llm
            items = extract_llm(question, answer, provenance, today) if have_llm() else extract(question, answer, provenance, today)
        except Exception: items = extract(question, answer, provenance, today)
        written = []
        for tier, text in items:
            existing = [strip_tags(r["memory"]) for r in self._by_tier(tier)]
            if is_duplicate(text, existing): continue                     # dedup before every write
            written.append({"tier": tier, "text": text, "record": self.store.add_memory(text, tier, self.user_id, thread_id)})
        return written
    def write_async(self, question, answer, thread_id="", provenance=""):
        """Runs after the final response is produced, so the user never waits on memory extraction."""
        t = threading.Thread(target=lambda: self._safe(question, answer, thread_id, provenance), daemon=True, name="mem-write")
        PENDING.append(t); t.start(); return t
    def _safe(self, *a):
        try: self.write_turn(*a)
        except Exception as e: print(f"[memory] async write failed: {e!r}")
    def end_session(self, messages):
        """Post-session extraction pipeline. The system prompt and summary notes are not conversation, so they are skipped."""
        from core.summarizer import NOTE
        convo = [m for m in messages if getattr(m, "type", "human") != "system" and not str(getattr(m, "content", m.get("content", "") if isinstance(m, dict) else "")).startswith(NOTE)]
        return self.store.add(convo, self.user_id) if convo else []
    def forget(self, needle: str) -> int:
        gone = [r for r in self.store.get_all(self.user_id) if needle.lower() in r["memory"].lower()]
        for r in gone: self.store.delete(r["id"])
        return len(gone)
    def update(self, old_needle: str, new_text: str, tag: str = "[Semantic]"):
        """Mem0 appends instead of replacing; this wrapper forgets the obsolete fact, then stores the new one."""
        self.forget(old_needle)
        return self.store.add_raw(new_text, self.user_id, tag)
