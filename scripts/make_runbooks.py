"""Generate 2 sample runbook PDFs (prose + Python + error table) and 1 HTML runbook."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Preformatted, Table, TableStyle, Spacer
from reportlab.lib import colors
from core import config

ss = getSampleStyleSheet()
code_style = ParagraphStyle("code", fontName="Courier", fontSize=8, leading=10)

def build(path, title, sections):
    story = [Paragraph(title, ss["Title"])]
    for h, prose, code, table in sections:
        story.append(Paragraph(h, ss["Heading2"]))
        story.append(Paragraph(prose, ss["BodyText"]))
        if code: story += [Spacer(1, 6), Preformatted(code, code_style)]
        if table:
            t = Table(table, hAlign="LEFT"); t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("FONTSIZE", (0, 0), (-1, -1), 8)]))
            story += [Spacer(1, 6), t]
        story.append(Spacer(1, 10))
    SimpleDocTemplate(str(path), pagesize=A4).build(story)

RETRY = '''import time

def retry_payment(job, max_retries=3, base_delay=2):
    """Retry a failed payment job with exponential backoff."""
    for attempt in range(1, max_retries + 1):
        try:
            return job.run()
        except TimeoutError:
            if attempt == max_retries:
                raise
            time.sleep(base_delay ** attempt)
'''
RESYNC = '''def resync_inventory(client, sku_batch):
    """Re-push SKU quantities after a stale-cache error."""
    for sku in sku_batch:
        client.invalidate(sku)
        client.push(sku, client.read_source(sku))
'''

def main():
    config.RUNBOOK_DIR.mkdir(parents=True, exist_ok=True)
    build(config.RUNBOOK_DIR / "payment_worker_runbook.pdf", "Payment Worker Runbook", [
        ("Overview", "The payment worker processes card payments from the order queue. It retries transient failures and parks jobs that keep failing. On-call engineers use this runbook when alerts fire for the payment-worker service.", None, None),
        ("Error 5042: Gateway timeout", "Error 5042 means the payment gateway did not respond in time. First confirm the gateway status page, then check queue depth. If the gateway is healthy, replay the job with the retry script below. Never replay more than 3 times, because duplicate charges are possible without an idempotency key.", RETRY, None),
        ("Error code reference", "Use this table to map worker error codes to first actions.", None,
         [["Code", "Meaning", "First action"], ["5042", "Gateway timeout", "Retry with backoff"], ["5001", "Invalid card token", "Ask customer to re-enter card"], ["5017", "Duplicate charge guard", "Check idempotency key, do not retry"]]),
        ("Escalation", "If error 5042 persists for more than 15 minutes, page the payments on-call and open an incident. Roll back the latest worker release only after confirming the commit is the cause.", None, None)])
    build(config.RUNBOOK_DIR / "inventory_sync_runbook.pdf", "Inventory Sync Runbook", [
        ("Overview", "Inventory sync keeps warehouse stock and the storefront consistent. Most incidents come from stale caches or delayed queue consumers.", None, None),
        ("Error 6012: Stale cache", "Error 6012 means the storefront cache holds an older quantity than the source of truth. Run the resync script for the affected SKUs, then verify counts match in the dashboard.", RESYNC, None),
        ("Error code reference", "Inventory sync error codes.", None, [["Code", "Meaning", "First action"], ["6001", "Source unreachable", "Check warehouse API health"], ["6012", "Stale cache", "Run resync script"]]),
        ("On-call policy", "Inventory sync incidents are severity 2 unless checkout is blocked, in which case they become severity 1. Acknowledge pages within 10 minutes.", None, None)])
    (config.RUNBOOK_DIR / "order_api_runbook.html").write_text("""<html><head><title>Order API Runbook</title><style>.x{}</style></head><body>
<nav>Home | Docs | Login</nav><h1>Order API Runbook</h1>
<h2>Error 4290: Rate limited</h2><p>Error 4290 means the client exceeded the order API rate limit. Ask the client to back off for 60 seconds. Raise limits only with approval from the platform team.</p>
<h2>Error 4001: Bad request</h2><p>Validate the payload against the order schema and return a clear message to the caller.</p>
<script>track()</script><footer>Copyright Acme</footer></body></html>""")
    print("runbooks written to", config.RUNBOOK_DIR)

if __name__ == "__main__": main()
