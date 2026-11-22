"""Three-case router (+ a 'static' shortcut). Free rule check first; the LLM classifier only runs when no rule matches."""
import re

GREETING = re.compile(r"^\s*(hi|hello|hey|thanks|thank you|good (morning|afternoon|evening)|who are you|what can you (do|help))\b", re.I)
STATIC = re.compile(r"\b(sla policy|sla|service catalog|which services|who owns)\b", re.I)
LIVE_SLA = re.compile(r"\b(sla status|compliance|is .* meeting)\b", re.I)
OPS = re.compile(r"\b(failed|orders?|commits?|branch(es)?|prs?|pull requests?|alerts?|on[- ]?call|rollback|latency|database|db|git|blame|diff)\b", re.I)
RUNBOOK = re.compile(r"\b(error \d+|runbook|policy|retry script|escalat\w*|how do i fix|holiday|procedure|stipend|cafeteria)\b", re.I)

def route_by_rules(q: str) -> str | None:
    if GREETING.match(q): return "chat"
    if STATIC.search(q) and not LIVE_SLA.search(q) and not OPS.search(q): return "static"
    if OPS.search(q) or LIVE_SLA.search(q): return "ops"
    if RUNBOOK.search(q): return "runbook"
    return None
