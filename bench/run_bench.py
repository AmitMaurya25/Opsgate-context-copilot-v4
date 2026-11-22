"""Run every question in both modes -> bench/results.csv. Needs GROQ_API_KEY for meaningful numbers.
Categories: chat, runbook, out_of_scope, ops, long (multi-turn, triggers the summarizer), memory (two sessions, same user).
Usage: python bench/run_bench.py [--mode naive|engineered] [--cats chat,ops]"""
import argparse, json, sys, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.graph import OpsGate
from core.llm import IGNORANCE

FIELDS = ["question_id", "category", "mode", "input_tokens", "tools_in_prompt", "rag_tokens", "tool_output_tokens", "correct", "latency_s",
          "sent_tokens", "cached_tokens", "cache_hit_pct", "effective_tokens", "summary_saved", "route"]

def system_prompt(agent, thread):
    st = agent.graph.get_state({"configurable": {"thread_id": thread}}).values
    return st["messages"][0].content if st and st.get("messages") else ""

def correct(q, r, sysprompt=""):
    a = r["answer"].lower(); t = r["telemetry"]
    if q["category"] == "chat": return bool(a.strip()) and IGNORANCE.lower() not in a
    ok = q["expect"].lower() in (a + " " + sysprompt.lower() if q["category"] == "memory" else a)
    if q.get("forbid") and q["forbid"].lower() in sysprompt.lower(): ok = False
    return ok

def run_question(agent, q, mode):
    uid, thread = f"bench-{q['id']}-{mode}", f"bench-{mode}-{q['id']}"
    if q["category"] == "long":                                    # earlier turns in the SAME thread; only the last turn is measured
        for turn in q["setup"]: agent.ask(turn, mode, uid, thread)
    if q["category"] == "memory":
        agent.mem(uid).forget("")                                  # clean slate per question
        for turn in q["session1"]: agent.ask(turn, mode, uid, thread + "-s1")
        agent.end_session(uid, mode, thread + "-s1")
        if q.get("update"): agent.mem(uid).update(q["update"]["old"], q["update"]["new"])
        thread += "-s2"                                            # brand-new session
    r = agent.ask(q["q"], mode, uid, thread)
    return r, system_prompt(agent, thread)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mode"); ap.add_argument("--cats"); a = ap.parse_args()
    qs = json.loads((Path(__file__).parent / "questions.json").read_text())
    if a.cats: qs = [q for q in qs if q["category"] in a.cats.split(",")]
    agent, rows = OpsGate(), []
    for mode in ([a.mode] if a.mode else ["naive", "engineered"]):
        for q in qs:
            r, sp = run_question(agent, q, mode); t = r["telemetry"]
            rows.append({"question_id": q["id"], "category": q["category"], "mode": mode, "input_tokens": t["input_tokens"],
                         "tools_in_prompt": t["tools_in_prompt"], "rag_tokens": t["elements"]["rag"], "tool_output_tokens": t["elements"]["tool_io"],
                         "correct": int(correct(q, r, sp)), "latency_s": t["latency_s"], "sent_tokens": t["sent_tokens"], "cached_tokens": t["cached_tokens"],
                         "cache_hit_pct": t["cache_hit_pct"], "effective_tokens": t["effective_tokens"], "summary_saved": t["summary_saved"], "route": t["route"]})
            print(mode, q["id"], rows[-1]["input_tokens"], rows[-1]["correct"])
    with open(Path(__file__).parent / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    agent.host.close()
    import collections
    tot = collections.defaultdict(list)
    for r in rows: tot[r["mode"]].append(r["sent_tokens"])
    if len(tot) == 2:
        n, e = sum(tot["naive"]), sum(tot["engineered"]); print(f"\nTOKEN SAVINGS: naive {n:,} -> engineered {e:,} = {100 * (1 - e / max(n, 1)):.1f}% saved")

if __name__ == "__main__": main()
