"""Lossless compression. Applied to MCP / RAG output ONLY, never to user messages or history."""
import json, re

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
TAGS = re.compile(r"</?[A-Za-z][^>]*>")
TS = re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b")
DECOR = re.compile(r"^[\s\-=_*~+|#]{4,}$", re.M)
SPACES = re.compile(r"(?<=\S)[ \t]{2,}")     # only INNER runs: leading indentation (code, diffs) survives
BLANKS = re.compile(r"\n{3,}")

def strip_ansi(t): return ANSI.sub("", t)
def strip_tags(t): return TAGS.sub("", t)
def strip_timestamps(t): return TS.sub("", t)
def strip_decor(t): return DECOR.sub("", t)
def collapse_ws(t): return BLANKS.sub("\n\n", SPACES.sub(" ", t)).strip()
MIN_DEDUPE_LEN = 8        # short structural lines ("}", "end", "return") repeat legitimately and are never dropped
def dedupe_lines(t):
    seen, out = set(), []
    for ln in t.splitlines():
        if len(ln.strip()) >= MIN_DEDUPE_LEN and ln in seen: continue
        seen.add(ln); out.append(ln)
    return "\n".join(out)
def minify_json(t):
    try: return json.dumps(json.loads(t), separators=(",", ":"))
    except Exception: return t

STEPS = [strip_ansi, strip_tags, strip_timestamps, strip_decor, minify_json, collapse_ws, dedupe_lines]

def clean_context(text) -> str:
    if not isinstance(text, str): text = json.dumps(text, separators=(",", ":"), default=str)
    for fn in STEPS: text = fn(text)
    return text

def savings(raw: str, cleaned: str) -> float:
    return round(100 * (1 - len(cleaned) / max(1, len(raw))), 1)

def cap_tokens(text: str, max_tokens: int) -> tuple[str, int]:
    """Hard ceiling for anything entering the prompt. Cuts on a line boundary and says how much was dropped. Returns (text, dropped_tokens)."""
    from core.telemetry import count_tokens
    n = count_tokens(text)
    if n <= max_tokens: return text, 0
    cut = text[: int(len(text) * max_tokens / n)]
    cut = cut[: cut.rfind("\n")] if "\n" in cut else cut
    return cut + f"\n[truncated: ~{n - max_tokens} tokens omitted; narrow the query or use page/page_size]", n - max_tokens
