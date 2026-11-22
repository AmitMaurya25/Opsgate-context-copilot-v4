"""Programmatic tool calling: paginate, strip fields and tabulate OUTSIDE the LLM."""
import csv, io
from collections import Counter
from core import config

DEFAULT_KEEP = {"query_failed_orders": ["txn_id", "order_id", "service", "error_code", "ts"]}

def run_paged(host, tool: str, args: dict, page_size: int | None = None, max_rows: int = 500) -> list[dict]:
    size, rows, page = page_size or config.PAGE_SIZE, [], 1
    while len(rows) < max_rows:
        res = host.call(tool, {**args, "page": page, "page_size": size})
        if not isinstance(res, dict) or "rows" not in res: return res            # not a paged tool
        rows += res["rows"]
        if not res.get("more_pages_available"): break
        page += 1
    return rows[:max_rows]

def strip_fields(rows: list[dict], keep: list[str]) -> list[dict]:
    return [{k: r.get(k) for k in keep} for r in rows]

def to_csv(rows: list[dict]) -> str:
    if not rows: return ""
    buf = io.StringIO(); w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()), lineterminator="\n"); w.writeheader(); w.writerows(rows)
    return buf.getvalue().strip()

def run_programmatic_task(host, tool: str, args: dict, keep: list[str] | None = None, group_by: str | None = "service") -> dict:
    """Multi-page fetch + field filtering + compact CSV. Returns {text, raw_chars, rows}."""
    rows = run_paged(host, tool, args)
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
        return {"text": str(rows), "raw_chars": len(str(rows)), "rows": 0}
    raw_chars = len(str(rows))
    keep = keep or next((v for k, v in DEFAULT_KEEP.items() if tool.endswith(k)), None) or list(rows[0].keys())
    slim = strip_fields(rows, keep)
    text = to_csv(slim)
    if group_by and group_by in keep:             # summary FIRST: if the cap ever truncates the table, the answer-bearing line survives
        text = "counts by %s: %s\n\n%s" % (group_by, dict(Counter(r[group_by] for r in slim)), text)
    return {"text": text, "raw_chars": raw_chars, "rows": len(rows)}
