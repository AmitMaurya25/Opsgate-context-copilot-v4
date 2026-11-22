"""Chat threads: persistent per user (JSON on disk), titled automatically from what the thread is about.
No Streamlit / Chroma imports here, so it is cheap to unit-test."""
from __future__ import annotations
import json, re, time, uuid
from core import config

STOP = set("a an the and or but if then so of to in on at for from by with about into over is are was were be been am do does did can could would should will "
           "i me my we our you your it its this that these those what which who whom how why when where please show tell give need want like help check find "
           "get let make just some any all me us them there here also not no yes ok hi hello hey thanks thank summarise summarize summary explain describe read".split())

def heuristic_title(text: str, files: list[str] | None = None, max_words: int = 5) -> str:
    """Offline title: the first few content words (error codes, service names and numbers kept), Title Cased."""
    words, seen = [], set()
    for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-']*", text or ""):
        lw = w.lower().strip("-_'")
        if len(lw) < 2 or lw in STOP or lw in seen: continue
        seen.add(lw); words.append(w.strip("-_'"))
        if len(words) == max_words: break
    if len(words) < 2 and files:                              # "summarise this" -> name it after the file
        stem = re.sub(r"[_\-]+", " ", files[0].rsplit(".", 1)[0]).strip()
        words = (words + stem.split())[:max_words]
    if not words: return "New chat"
    return " ".join(w if (w.isupper() or any(c.isdigit() for c in w)) else w[:1].upper() + w[1:] for w in words)

def llm_title(text: str, files: list[str] | None, answer: str = "") -> str | None:
    try:
        from core.llm import have_llm, get_llm
        if not have_llm(): return None
        prompt = ("Write a chat title of 3 to 5 words that names what this conversation is about. Be specific (use error codes, service names or file names "
                  "when present). No quotes, no trailing punctuation. Output only the title.\n\n"
                  f"<user_messages>\n{text[:600]}\n</user_messages>\n" + (f"<files>{', '.join(files)}</files>\n" if files else "") + (f"<assistant_reply>\n{answer[:300]}\n</assistant_reply>" if answer else ""))
        t = get_llm(config.SUMMARIZER_MODEL).invoke(prompt).content.strip().strip("\"'`*# ").splitlines()[0].rstrip(".:;,")
        return " ".join(t.split()[:7]) or None
    except Exception: return None

def unique_title(title: str, taken: list[str]) -> str:
    low = {t.lower() for t in taken}
    if title.lower() not in low: return title
    n = 2
    while f"{title} ({n})".lower() in low: n += 1
    return f"{title} ({n})"

def bucket(ts: float, now: float | None = None) -> str:
    age_days = ((now or time.time()) - ts) / 86400
    return "Today" if age_days < 1 else "Yesterday" if age_days < 2 else "Previous 7 days" if age_days < 7 else "Older"
BUCKETS = ["Today", "Yesterday", "Previous 7 days", "Older"]

class ThreadStore:
    def __init__(self, user_id: str, root=None):
        root = root or config.THREADS_DIR; root.mkdir(parents=True, exist_ok=True)
        self.path = root / (re.sub(r"[^\w\-]", "_", user_id or "anon") + ".json")
        try: self.data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        except Exception: self.data = {}                       # a corrupt file must never block the UI
    def _save(self): self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")
    def list(self) -> list[dict]: return sorted(self.data.values(), key=lambda t: t["updated"], reverse=True)
    def get(self, tid): return self.data.get(tid)
    def titles(self, exclude=None) -> list[str]: return [t["title"] for t in self.data.values() if t["id"] != exclude]
    def create(self) -> str:
        tid, now = uuid.uuid4().hex[:8], time.time()
        self.data[tid] = {"id": tid, "title": "New chat", "auto": True, "created": now, "updated": now, "gen": 0, "log": []}; self._save(); return tid
    def append(self, tid, role, text, files=None, extra=None):
        """`extra` carries a structured payload (the Naive-vs-OpsGate comparison) next to the plain text."""
        t = self.data[tid]; m = {"role": role, "text": text, "files": files or []}
        if extra: m["extra"] = extra
        t["log"].append(m); t["updated"] = time.time(); self._save()
    def user_turns(self, tid) -> int: return sum(1 for m in self.data[tid]["log"] if m["role"] == "user")
    def rename(self, tid, title, manual=True):
        title = " ".join((title or "").split())[:60] or "New chat"
        self.data[tid]["title"] = unique_title(title, self.titles(exclude=tid)) if title != "New chat" else title
        if manual: self.data[tid]["auto"] = False
        self._save()
    def auto_title(self, tid, answer: str = ""):
        """Title from the content: after the 1st message, and refined once after the 3rd (when there is more to go on). Manual renames are never overwritten."""
        t = self.data[tid]
        if not t["auto"]: return
        turns = self.user_turns(tid)
        if turns not in (1, 3): return
        users = [m for m in t["log"] if m["role"] == "user"]
        text = " ".join(m["text"] for m in users if not m["text"].startswith("(attached"))
        files = [f for m in users for f in m.get("files", [])]
        title = llm_title(text, files, answer) or heuristic_title(text, files)
        self.rename(tid, title, manual=False)
    def clear(self, tid): t = self.data[tid]; t["log"] = []; t["gen"] += 1; t["updated"] = time.time(); self._save()
    def delete(self, tid): self.data.pop(tid, None); self._save()
