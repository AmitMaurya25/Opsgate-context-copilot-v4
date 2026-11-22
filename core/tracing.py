"""Optional LangSmith tracing. OFF unless LANGSMITH_API_KEY is in .env, so with no key the app behaves exactly as before.

Privacy: by default (LANGSMITH_REDACT=1) prompts, tool inputs and outputs are NOT sent to LangSmith, only the run structure
(which node/tool ran, in what order, how long, errors). Set LANGSMITH_REDACT=0 only on non-sensitive demo data.
Tracing never touches the prompt, so the KV-cache-friendly prompt layout is unchanged.
"""
import hashlib, os

def setup_tracing() -> dict:
    if not os.getenv("LANGSMITH_API_KEY"):
        return {"enabled": False, "redacted": True, "project": ""}
    redact = os.getenv("LANGSMITH_REDACT", "1") != "0"
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault("LANGSMITH_PROJECT", "opsgate-context-copilot")
    os.environ.setdefault("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    os.environ["LANGSMITH_HIDE_INPUTS"] = "true" if redact else "false"
    os.environ["LANGSMITH_HIDE_OUTPUTS"] = "true" if redact else "false"
    return {"enabled": True, "redacted": redact, "project": os.environ["LANGSMITH_PROJECT"]}

def trace_config(mode: str, user_id: str, thread_id: str) -> dict:
    """Run name, tags and metadata for graph.invoke. User ids are hashed so no identity leaves the machine."""
    return {"run_name": f"opsgate.{mode}", "tags": [mode],
            "metadata": {"mode": mode, "user": hashlib.md5(user_id.encode()).hexdigest()[:8], "thread": thread_id}}

# ---------- v4: audit log + OpenTelemetry spans (Langfuse-compatible) ----------
import json, time, threading
from contextlib import contextmanager
from core import config

_audit_lock = threading.Lock()
def audit(event: dict):
    """Append one JSON line to logs/audit.jsonl. Everything is PII/secret-redacted BEFORE it touches disk."""
    from core.guardrails import redact
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **event}
    line = redact(json.dumps(rec, default=str, ensure_ascii=False))
    try:
        with _audit_lock:
            config.LOG_DIR.mkdir(parents=True, exist_ok=True)
            with open(config.AUDIT_LOG, "a", encoding="utf-8") as f: f.write(line + "\n")
    except Exception: pass                                  # auditing must never break a request

_tracer = {"obj": None, "mode": "off"}
def setup_otel() -> dict:
    """OpenTelemetry spans. With LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY the spans are exported to Langfuse (OTLP/HTTP); otherwise they stay
    local (the audit log is still written). Needs the optional packages opentelemetry-sdk + opentelemetry-exporter-otlp-proto-http."""
    if _tracer["obj"] is not None: return {"mode": _tracer["mode"]}
    import os, base64
    try:
        from opentelemetry import trace
        pk, sk = os.getenv("LANGFUSE_PUBLIC_KEY"), os.getenv("LANGFUSE_SECRET_KEY")
        if pk and sk:
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com").rstrip("/")
            auth = base64.b64encode(f"{pk}:{sk}".encode()).decode()
            prov = TracerProvider(); prov.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{host}/api/public/otel/v1/traces", headers={"Authorization": f"Basic {auth}"})))
            trace.set_tracer_provider(prov); _tracer["mode"] = "langfuse"
        else: _tracer["mode"] = "local"
        _tracer["obj"] = trace.get_tracer("opsgate")
    except Exception: _tracer["obj"], _tracer["mode"] = False, "off"
    return {"mode": _tracer["mode"]}

@contextmanager
def span(name: str, **attrs):
    """Context manager: an OTel span when available, a no-op otherwise. Attributes are redacted strings/numbers only."""
    setup_otel()
    if not _tracer["obj"]: yield None; return
    from core.guardrails import redact
    with _tracer["obj"].start_as_current_span(name) as sp:
        for k, v in attrs.items(): sp.set_attribute(k, v if isinstance(v, (int, float, bool)) else redact(str(v))[:300])
        yield sp

def traced(name: str):
    """Decorator for graph nodes: one span per node, state mode/role as attributes."""
    def deco(fn):
        def wrapper(s):
            with span(f"opsgate.{name}", mode=s.get("mode", ""), role=s.get("user_role", "")): return fn(s)
        wrapper.__name__ = fn.__name__; return wrapper
    return deco
