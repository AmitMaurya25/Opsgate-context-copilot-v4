"""Create data/mock_ops.db and data/mock_git.json (realistic, deterministic)."""
import json, random, sqlite3, sys
from datetime import datetime, timedelta
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import config

random.seed(42)
SERVICES = config.SERVICES
ERRORS = {"payment-worker": ["5042", "5001", "5017"], "order-api": ["4001", "4290", "5003"], "inventory-sync": ["6001", "6012"]}
now = datetime(2026, 11, 19, 10, 0, 0)

def main():
    config.DATA.mkdir(exist_ok=True)
    if config.DB_PATH.exists(): config.DB_PATH.unlink()
    db = sqlite3.connect(config.DB_PATH)
    db.executescript("""
    CREATE TABLE orders(order_id INTEGER PRIMARY KEY, customer TEXT, amount REAL, status TEXT, service TEXT, created_at TEXT);
    CREATE TABLE failed_txns(txn_id INTEGER PRIMARY KEY, order_id INTEGER, service TEXT, error_code TEXT, message TEXT, ts TEXT,
        trace_id TEXT, internal_uuid TEXT, raw_headers TEXT, node TEXT, region TEXT, retry_count INTEGER, payload_size INTEGER,
        client_ip TEXT, user_agent TEXT, build TEXT, pod TEXT, queue TEXT, latency_ms INTEGER, upstream TEXT,
        cache_hit INTEGER, auth_scope TEXT, correlation_id TEXT, shard INTEGER, schema_ver TEXT);
    CREATE TABLE latency_stats(service TEXT, p50 INTEGER, p95 INTEGER, p99 INTEGER, error_rate REAL, window TEXT);
    """)
    for i in range(1, 501):
        db.execute("INSERT INTO orders VALUES(?,?,?,?,?,?)", (i, f"cust_{random.randint(1,120)}", round(random.uniform(5, 900), 2),
            random.choice(["paid", "failed", "pending", "refunded"]), random.choice(SERVICES), (now - timedelta(minutes=i * 7)).isoformat()))
    for i in range(1, 201):
        svc = random.choices(SERVICES, weights=[6, 3, 1])[0]; code = random.choice(ERRORS[svc])
        db.execute("INSERT INTO failed_txns VALUES(" + ",".join("?" * 25) + ")", (i, random.randint(1, 500), svc, code,
            f"{svc} failed with error {code}", (now - timedelta(minutes=i * 5)).isoformat(),
            f"trace-{random.getrandbits(48):x}", f"uuid-{random.getrandbits(64):x}", "X-Req-Id: abc; X-Fwd: 10.0.0.1; Accept: */*",
            f"node-{random.randint(1,9)}", random.choice(["ap-south-1", "eu-west-1"]), random.randint(0, 4), random.randint(200, 9000),
            f"10.1.{random.randint(0,255)}.{random.randint(1,254)}", "python-requests/2.31", f"build-{random.randint(100,140)}",
            f"pod-{random.randint(1,20)}", random.choice(["q-high", "q-low"]), random.randint(20, 2500), "gateway.internal",
            random.randint(0, 1), "orders:write", f"corr-{random.getrandbits(40):x}", random.randint(0, 7), "v3"))
    for s in SERVICES:
        db.execute("INSERT INTO latency_stats VALUES(?,?,?,?,?,?)", (s, random.randint(40, 90), random.randint(200, 400), random.randint(600, 1200), round(random.uniform(0.2, 4.5), 2), "24h"))
    db.commit(); db.close()
    git = {}
    for s in SERVICES:
        commits = [{"sha": f"{random.getrandbits(28):07x}", "author": random.choice(["asha", "ravi", "meera"]),
                    "message": random.choice(["fix retry backoff", "bump timeout", "add idempotency key", "refactor queue consumer", "patch error 5042 handler"]),
                    "ts": (now - timedelta(hours=i * 9)).isoformat(), "files": random.randint(1, 8)} for i in range(12)]
        git[s] = {"branches": ["main", "release/2.4", "hotfix/retry", "feature/queue-v2"], "commits": commits,
                  "prs": [{"id": 100 + i, "title": f"PR {100+i}: {random.choice(['retry policy','metrics','cleanup'])}", "state": "open"} for i in range(8)],
                  "diffs": {c["sha"]: "--- a/worker.py\n+++ b/worker.py\n@@ -10,3 +10,4 @@\n- retries = 1\n+ retries = 3\n+ backoff = 2" for c in commits}}
    config.GIT_JSON.write_text(json.dumps(git, indent=1))
    print("seeded", config.DB_PATH, config.GIT_JSON)

if __name__ == "__main__": main()
