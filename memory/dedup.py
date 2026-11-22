"""Temporal conflict resolution. LLM merges conflicts keeping the most recent; rule-based fallback offline."""
import re
from core.llm import get_llm, have_llm
from core import config
from memory.mem0_store import strip_tags

def _norm(t): return re.sub(r"\W+", " ", strip_tags(t).lower()).strip()

def dedupe_rules(records: list[dict]) -> list[dict]:
    """Keep the newest record per normalised text."""
    latest = {}
    for r in sorted(records, key=lambda r: r.get("ts", 0)): latest[_norm(r["memory"])] = r
    return list(latest.values())

def dedupe_llm(records: list[dict]) -> list[str]:
    lines = "\n".join(f"{i+1}. ({r.get('ts', 0)}) {strip_tags(r['memory'])}" for i, r in enumerate(sorted(records, key=lambda r: r.get('ts', 0))))
    prompt = ("Merge duplicate or conflicting memories. When two conflict, keep only the one with the later timestamp. "
              f"Return one memory per line, no numbering, no commentary.\n<memories>\n{lines}\n</memories>")
    out = get_llm(config.SUMMARIZER_MODEL).invoke(prompt).content
    return [l.strip("-• ").strip() for l in out.splitlines() if l.strip()]

def dedupe(records: list[dict]) -> list[str]:
    if have_llm() and len(records) > 1: return dedupe_llm(records)
    return [strip_tags(r["memory"]) for r in dedupe_rules(records)]

def is_duplicate(text: str, existing: list[str], threshold: float = 0.85) -> bool:
    """True when `text` repeats (or nearly repeats) a stored memory: normalised equality or token Jaccard >= threshold."""
    a = set(_norm(text).split())
    for e in existing:
        b = set(_norm(e).split())
        if a and b and (a == b or len(a & b) / len(a | b) >= threshold): return True
    return False
