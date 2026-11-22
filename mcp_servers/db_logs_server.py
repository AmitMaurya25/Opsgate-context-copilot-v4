"""MCP server: orders and failed transactions DB (5 tools). query_failed_orders returns BULKY raw JSON on purpose."""
import sqlite3, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mcp.server.fastmcp import FastMCP
from core import config

mcp = FastMCP("db-logs")
def _q(sql, args=()):
    db = sqlite3.connect(config.DB_PATH); db.row_factory = sqlite3.Row
    rows = [dict(r) for r in db.execute(sql, args)]; db.close(); return rows

@mcp.tool()
def query_failed_orders(limit: int = 50, page: int = 1, page_size: int = 50) -> dict:
    """Query failed order transactions from the database with all raw fields (service, error_code, trace ids, headers), paginated."""
    total = _q("SELECT COUNT(*) c FROM failed_txns")[0]["c"]
    limit = min(limit, total); size = min(page_size, limit); off = (page - 1) * size
    rows = _q("SELECT * FROM failed_txns ORDER BY ts DESC LIMIT ? OFFSET ?", (max(0, min(size, limit - off)), off))
    return {"rows": rows, "page": page, "total": limit, "more_pages_available": off + size < limit}

@mcp.tool()
def get_order(order_id: int) -> dict:
    """Get one customer order by its order id."""
    r = _q("SELECT * FROM orders WHERE order_id=?", (order_id,)); return r[0] if r else {"error": "not found"}

@mcp.tool()
def count_errors_by_service(hours: int = 24) -> list[dict]:
    """Count failed transactions grouped by service and error code."""
    return _q("SELECT service, error_code, COUNT(*) AS n FROM failed_txns GROUP BY service, error_code ORDER BY n DESC")

@mcp.tool()
def get_latency_stats(service: str) -> dict:
    """Get latency percentiles (p50, p95, p99) and error rate of a service."""
    r = _q("SELECT * FROM latency_stats WHERE service=?", (service,)); return r[0] if r else {"error": "unknown service"}

@mcp.tool()
def describe_schema() -> dict:
    """Describe the database tables and columns."""
    return {t["name"]: [c["name"] for c in _q(f"PRAGMA table_info({t['name']})")] for t in _q("SELECT name FROM sqlite_master WHERE type='table'")}

if __name__ == "__main__": mcp.run(transport="stdio")
