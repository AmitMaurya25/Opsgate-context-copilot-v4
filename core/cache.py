"""KV-cache awareness.

LLM providers reuse the attention (KV) state of a prompt PREFIX that they have already seen. Everything after the first changed
token is recomputed (and billed at the full rate). So the rule is: static on top, dynamic at the bottom, never reorder.

`PrefixCache` estimates the hit rate: for each prompt it finds the longest token prefix shared with any recent prompt and
rounds it down to whole cache blocks. When the provider reports real cached tokens (usage_metadata) the app shows those instead.
"""
import json, re
from collections import deque
from core import config

_enc = None
def _tokens(text: str) -> list:
    global _enc
    try:
        if _enc is None:
            import tiktoken; _enc = tiktoken.get_encoding("cl100k_base")
        return _enc.encode(text)
    except Exception:
        return re.findall(r"\s*\S+", text)           # offline approximation: word-ish pieces

def render_prompt(messages, tools=None) -> str:
    """Provider-like rendering: tool schemas first (stable order), then every message in order."""
    parts = []
    for t in tools or []:
        parts.append("TOOL " + json.dumps(getattr(t, "name", str(t))) + " " + (getattr(t, "description", "") or ""))
    for m in messages if isinstance(messages, list) else [messages]:
        content = m if isinstance(m, str) else (m.content if isinstance(m.content, str) else json.dumps(m.content, default=str))
        calls = getattr(m, "tool_calls", None)
        parts.append(f"{getattr(m, 'type', 'human')}: {content}" + (" " + json.dumps(calls, sort_keys=True, default=str) if calls else ""))
    return "\n".join(parts)

class PrefixCache:
    def __init__(self, keep: int = 24, block: int | None = None):
        self.seen: deque = deque(maxlen=keep); self.block = block or config.CACHE_BLOCK_TOKENS
    def observe(self, prompt_text: str) -> tuple[int, int]:
        """Returns (total_tokens, estimated_cached_tokens) and remembers the prompt."""
        toks = _tokens(prompt_text); best = 0
        for old in self.seen:
            n = 0
            for a, b in zip(old, toks):
                if a != b: break
                n += 1
            best = max(best, n)
        self.seen.append(toks)
        return len(toks), (best // self.block) * self.block
