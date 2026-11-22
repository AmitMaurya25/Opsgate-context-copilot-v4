"""Governance layer: RBAC role gate, PII / secret redaction, indirect prompt-injection scanner."""
import re

READ_ONLY, PRIVILEGED_WRITE = "READ_ONLY", "PRIVILEGED_WRITE"
ROLES = ["Viewer", "OnCall-Engineer", "Admin"]
ROLE_RISKS = {"Viewer": {READ_ONLY}, "OnCall-Engineer": {READ_ONLY, PRIVILEGED_WRITE}, "Admin": {READ_ONLY, PRIVILEGED_WRITE}}

def normalize_role(role: str | None) -> str:
    r = (role or "").strip().lower().replace("_", "-").replace(" ", "-")
    return next((x for x in ROLES if x.lower() == r), "Viewer")      # unknown role = least privilege

def role_allows(role: str, risk: str) -> bool: return risk in ROLE_RISKS[normalize_role(role)]
def needs_approval(risk: str) -> bool: return risk == PRIVILEGED_WRITE

# ---- PII / secrets ----
SECRETS = [
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b"), "[REDACTED_API_KEY]"),
    (re.compile(r"\bgsk_[A-Za-z0-9]{16,}\b"), "[REDACTED_API_KEY]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED_AWS_KEY]"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9\-._~+/]{12,}=*"), "Bearer [REDACTED_TOKEN]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"), "[REDACTED_EMAIL]"),
]
def redact(text) -> str:
    if not isinstance(text, str): text = str(text)
    for rx, repl in SECRETS: text = rx.sub(repl, text)
    return text

# ---- indirect prompt injection ----
INJECTION = re.compile(
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions?|prompts?|rules?)|disregard (all |any |the )?(previous|prior|above) (instructions?|prompts?)"
    r"|system prompt|you are now\b|new instructions?:|exfiltrate|reveal (your|the) (system|hidden)|act as (an? )?(admin|root|developer)", re.I)
QUARANTINE = "[quarantined: suspected prompt injection removed]"

def scan_injection(text: str) -> tuple[str, list[str]]:
    """Quarantine whole lines that look like instruction overrides. Returns (clean_text, findings)."""
    out, found = [], []
    for ln in str(text).splitlines():
        if INJECTION.search(ln): found.append(ln.strip()[:120]); out.append(QUARANTINE)
        else: out.append(ln)
    return "\n".join(out), found

# tool outputs from these sources are untrusted free text: always scanned
UNTRUSTED_SERVERS = ("web_search", "db_logs")
def sanitize_tool_output(text: str, server: str = "") -> tuple[str, list[str]]:
    findings: list[str] = []
    if server in UNTRUSTED_SERVERS or not server: text, findings = scan_injection(text)
    return redact(text), findings
