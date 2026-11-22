"""MCP server: incidents and on-call (7 tools) + 2 resources + 2 prompts. trigger_rollback, restart_worker and merge_pull_request are PRIVILEGED_WRITE (see mcp_servers.json)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("incident-ops")
ONCALL = {"payment-worker": "asha", "order-api": "ravi", "inventory-sync": "meera"}
SLA = "SLA policy: Sev1 ack within 5 minutes, Sev2 within 10 minutes, Sev3 next business day. Payments and checkout are always Sev1 when blocked."
CATALOG = "Services: payment-worker (payments team), order-api (orders team), inventory-sync (supply team)."

@mcp.tool()
def list_active_alerts() -> list[dict]:
    """List currently firing production alerts."""
    return [{"service": "payment-worker", "alert": "error 5042 spike", "severity": "Sev1"}, {"service": "inventory-sync", "alert": "queue lag", "severity": "Sev2"}]

@mcp.tool()
def get_oncall(service: str) -> dict:
    """Get the on-call engineer for a service."""
    return {"service": service, "oncall": ONCALL.get(service, "unknown")}

@mcp.tool()
def trigger_rollback(service: str, version: str) -> dict:
    """Roll back a service to a previous version (mock, returns a confirmation only)."""
    return {"status": "rollback_requested", "service": service, "version": version}

@mcp.tool()
def restart_worker(service: str) -> dict:
    """Restart the worker processes of a service (mock, returns a confirmation only)."""
    return {"status": "restart_requested", "service": service}

@mcp.tool()
def merge_pull_request(service: str, pr_id: int) -> dict:
    """Merge an open pull request into the main branch of a service (mock, returns a confirmation only)."""
    return {"status": "merge_requested", "service": service, "pr_id": pr_id}

@mcp.tool()
def draft_incident_email(service: str, summary: str) -> str:
    """Draft an incident notification email for a service."""
    return f"Subject: Incident on {service}\n\nSummary: {summary}\nOn-call: {ONCALL.get(service, 'unknown')}"

@mcp.tool()
def get_sla_status(service: str) -> dict:
    """Get SLA compliance status of a service."""
    return {"service": service, "sla_met": service != "payment-worker", "window": "24h"}

@mcp.resource("service://sla-policy")
def sla_policy() -> str: return SLA

@mcp.resource("service://service-catalog")
def service_catalog() -> str: return CATALOG

@mcp.prompt()
def postmortem_template(service: str) -> str:
    return f"Write a blameless postmortem for {service}: timeline, impact, root cause, fixes, follow-ups."

@mcp.prompt()
def incident_summary(service: str) -> str:
    return f"Summarize the current incident on {service} in 5 bullet points for executives."

if __name__ == "__main__": mcp.run(transport="stdio")
