"""Token counting and per-context-element breakdown (history, prompt, instructions, schemas, tool I/O, RAG, memory)."""
import json, time
from functools import lru_cache
from dataclasses import dataclass, field

_enc = None
def count_tokens(text) -> int:
    if not isinstance(text, str): text = json.dumps(text, default=str)
    return _count(text)

@lru_cache(maxsize=4096)                 # the same schemas / system prompt are counted over and over
def _count(text: str) -> int:
    global _enc
    try:
        if _enc is None:
            import tiktoken; _enc = tiktoken.get_encoding("cl100k_base")
        return len(_enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)      # offline fallback approximation

ELEMENTS = ["history", "prompt", "instructions", "tool_schemas", "tool_io", "rag", "memory"]

@dataclass
class Telemetry:
    mode: str = ""
    elements: dict = field(default_factory=lambda: {e: 0 for e in ELEMENTS})
    tools_in_prompt: int = 0
    total_tools: int = 0
    top_score: float | None = None
    chunks_kept: int = 0
    cleaner_saved: int = 0
    summary_saved: int = 0
    llm_calls: int = 0
    sent_tokens: int = 0            # prompt tokens sent over ALL llm calls of this run (tool loops re-send the context)
    cached_tokens: int = 0          # of those, tokens served from the KV cache (provider-reported, else estimated)
    cache_source: str = "estimate"  # "provider" when usage_metadata carried cache_read
    prefix_tokens: int = 0          # static prefix (system + tools) of the first call
    provider_input: int = 0
    route: str = ""
    route_reason: str = ""                                    # rule | llm | naive
    tool_calls: list = field(default_factory=list)            # [{n, tool, args, ms, out_tokens, status, preview}]
    llm_log: list = field(default_factory=list)               # [{n, sent, cached, ms}]
    chunks: list = field(default_factory=list)                # [{section, kind, score, tokens}]
    truncated_tokens: int = 0
    trace: list = field(default_factory=list)                 # ["Memory recall", "RAG miss (0.18)", "Deferred search", "utility_tools_convert_currency"]
    provenance: str = ""                                      # runbook | live_mcp | ignorance | chat | upload | naive
    output_tokens: int = 0
    contexts: list = field(default_factory=list)              # text actually given to the LLM as evidence (for RAGAS)
    memory_used: dict = field(default_factory=dict)           # tier -> hits injected
    role: str = ""
    t0: float = field(default_factory=time.time)
    latency_s: float = 0.0
    def add(self, element: str, text): self.elements[element] += count_tokens(text)
    @property
    def input_tokens(self) -> int: return sum(self.elements.values())
    @property
    def cache_hit_pct(self) -> float: return round(100 * self.cached_tokens / self.sent_tokens, 1) if self.sent_tokens else 0.0
    @property
    def effective_tokens(self) -> int: return max(0, self.sent_tokens - self.cached_tokens)
    def log_tool(self, name, args, out, ms, status="ok"):
        """Local-only detail for the UI (previews are short; nothing here goes to LangSmith)."""
        self.tool_calls.append({"n": len(self.tool_calls) + 1, "tool": name, "args": json.dumps(args, default=str)[:160], "ms": round(ms),
                                "out_tokens": count_tokens(out), "status": status, "preview": out[:240]})
    def record_call(self, total: int, cached_est: int, usage: dict | None = None):
        """Provider usage wins over estimates when present."""
        self.sent_tokens += total; self.cached_tokens += cached_est
        if usage and usage.get("input_tokens"):
            self.provider_input += usage["input_tokens"]
            cr = (usage.get("input_token_details") or {}).get("cache_read")
            if cr is not None: self.cached_tokens += cr - cached_est; self.cache_source = "provider"
    def finish(self): self.latency_s = round(time.time() - self.t0, 3); return self
    def as_dict(self):
        return {"mode": self.mode, "route": self.route, "input_tokens": self.input_tokens, "elements": self.elements,
                "tools_in_prompt": self.tools_in_prompt, "total_tools": self.total_tools, "top_score": self.top_score,
                "chunks_kept": self.chunks_kept, "cleaner_saved": self.cleaner_saved, "summary_saved": self.summary_saved,
                "llm_calls": self.llm_calls, "latency_s": self.latency_s, "sent_tokens": self.sent_tokens,
                "cached_tokens": self.cached_tokens, "cache_hit_pct": self.cache_hit_pct, "effective_tokens": self.effective_tokens,
                "cache_source": self.cache_source, "route_reason": self.route_reason, "tool_calls": self.tool_calls, "llm_log": self.llm_log,
                "chunks": self.chunks, "truncated_tokens": self.truncated_tokens, "prefix_tokens": self.prefix_tokens, "provider_input": self.provider_input,
                "trace": self.trace, "provenance": self.provenance, "output_tokens": self.output_tokens, "contexts": self.contexts, "memory_used": self.memory_used, "role": self.role,
                "latency_ms": round(self.latency_s * 1000)}
