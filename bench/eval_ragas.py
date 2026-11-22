"""Side-by-side evaluation of the Naive baseline vs the OpsGate pipeline on bench/questions.json.
Metrics: Faithfulness, Answer Relevancy, Context Precision (+ measured token usage). Writes bench/results.json, results_summary.md, results_summary.png.

Scorer: real RAGAS (LLM judge) when `ragas` is installed AND GROQ_API_KEY is set; otherwise a transparent local heuristic. The report says which one ran.
Usage: python bench/eval_ragas.py [--limit 20] [--cats chat,runbook,out_of_scope,ops,long] [--judge auto|ragas|heuristic]"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from core.graph import OpsGate
from core.llm import have_llm, get_embedder, cosine
from run_bench import run_question

STOP = set("the a an of to in on for and or is are was were be do does did how what which who when where why can could would should i me my we our you your it its this that with from by at as about into please".split())
def _toks(t): return {w for w in re.findall(r"[a-z0-9]+", (t or "").lower()) if w not in STOP}
REFUSAL = ("cannot verify", "cannot answer", "could not find")

def heuristic(q, r):
    """faithfulness: share of answer content words supported by the evidence; relevancy: embedding cosine(question, answer);
    context precision: share of retrieved contexts that contain the expected fact (1.0 when nothing was needed and nothing was retrieved)."""
    ans, ctx = re.sub(r"^\[[^\]]+\]\s*", "", r["answer"]), r["telemetry"].get("contexts") or []
    a = _toks(ans); evidence = _toks(" ".join(ctx) + " " + q["q"])
    refused = any(x in ans.lower() for x in REFUSAL)
    faith = np.nan if q["category"] == "chat" else (1.0 if refused else (len(a & evidence) / len(a) if a else 0.0))
    emb = get_embedder(); rel = max(0.0, cosine(emb.embed([q["q"]])[0], emb.embed([ans or " "])[0]))
    exp = (q.get("expect") or "").lower()
    if q["category"] in ("chat", "out_of_scope", "long") or not exp: prec = np.nan
    else: prec = (sum(exp in c.lower() for c in ctx) / len(ctx)) if ctx else 0.0
    return {"faithfulness": faith, "answer_relevancy": rel, "context_precision": prec}

def ragas_scores(rows):
    from ragas import evaluate, EvaluationDataset
    from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithoutReference
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_huggingface import HuggingFaceEmbeddings
    from core.llm import get_llm
    from core import config
    ds = EvaluationDataset.from_list([{"user_input": x["q"], "response": x["answer"], "retrieved_contexts": x["contexts"] or [""]} for x in rows])
    res = evaluate(ds, metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithoutReference()], llm=LangchainLLMWrapper(get_llm()),
                   embeddings=LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL))).to_pandas()
    return [{"faithfulness": float(r["faithfulness"]), "answer_relevancy": float(r["answer_relevancy"]), "context_precision": float(r["llm_context_precision_without_reference"])} for _, r in res.iterrows()]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=20); ap.add_argument("--cats", default="chat,runbook,out_of_scope,ops,long"); ap.add_argument("--judge", default="auto"); a = ap.parse_args()
    qs = [q for q in json.loads((HERE / "questions.json").read_text()) if q["category"] in a.cats.split(",")][: a.limit]
    agent, rows = OpsGate(), []
    for mode in ("naive", "engineered"):
        for q in qs:
            r, _ = run_question(agent, q, mode); t = r["telemetry"]
            rows.append({"id": q["id"], "category": q["category"], "mode": mode, "q": q["q"], "answer": r["answer"], "contexts": t.get("contexts") or [], "telemetry": {"contexts": t.get("contexts") or []},
                         "input_tokens": t["sent_tokens"], "output_tokens": t.get("output_tokens", 0), "latency_ms": round(t["latency_s"] * 1000), "tools": t["tools_in_prompt"]})
            print(mode, q["id"], rows[-1]["input_tokens"])
    agent.host.close()
    scorer = "heuristic (local fallback, NOT RAGAS)"
    if a.judge in ("auto", "ragas") and have_llm():
        try:
            for m in ("naive", "engineered"):
                sub = [x for x in rows if x["mode"] == m]
                for x, s in zip(sub, ragas_scores(sub)): x["scores"] = s
            scorer = "RAGAS (LLM judge)"
        except Exception as e: print(f"[eval] RAGAS unavailable ({type(e).__name__}: {str(e)[:100]}); using heuristic")
    qmap = {q["id"]: q for q in qs}
    for x in rows:
        if "scores" not in x: x["scores"] = heuristic(qmap[x["id"]], {"answer": x["answer"], "telemetry": x["telemetry"]})
    def agg(mode, key): return float(np.nanmean([x["scores"][key] for x in rows if x["mode"] == mode] or [np.nan]))
    def tok(mode, key): return float(np.mean([x[key] for x in rows if x["mode"] == mode]))
    S = {m: {"faithfulness": agg(m, "faithfulness"), "answer_relevancy": agg(m, "answer_relevancy"), "context_precision": agg(m, "context_precision"),
             "avg_input_tokens": tok(m, "input_tokens"), "avg_output_tokens": tok(m, "output_tokens"), "avg_latency_ms": tok(m, "latency_ms"), "avg_tools": tok(m, "tools")} for m in ("naive", "engineered")}
    saved = 100 * (1 - S["engineered"]["avg_input_tokens"] / max(S["naive"]["avg_input_tokens"], 1))
    (HERE / "results.json").write_text(json.dumps({"scorer": scorer, "summary": S, "rows": [{k: v for k, v in x.items() if k not in ("contexts", "telemetry")} for x in rows]}, indent=1, default=float))
    f = lambda v: "n/a" if v != v else f"{v:.2f}"
    md = f"""# OpsGate vs Naive: evaluation summary

Questions: {len(qs)} (categories: {a.cats}) · Scorer: **{scorer}** · LLM: {'Groq' if have_llm() else 'offline fake (numbers are NOT meaningful for quality)'}

| Metric | Naive baseline | OpsGate |
|---|---|---|
| Faithfulness | {f(S['naive']['faithfulness'])} | {f(S['engineered']['faithfulness'])} |
| Answer relevancy | {f(S['naive']['answer_relevancy'])} | {f(S['engineered']['answer_relevancy'])} |
| Context precision | {f(S['naive']['context_precision'])} | {f(S['engineered']['context_precision'])} |
| Avg input tokens / question | {S['naive']['avg_input_tokens']:,.0f} | {S['engineered']['avg_input_tokens']:,.0f} |
| Avg output tokens / question | {S['naive']['avg_output_tokens']:,.0f} | {S['engineered']['avg_output_tokens']:,.0f} |
| Avg tools in prompt | {S['naive']['avg_tools']:.1f} | {S['engineered']['avg_tools']:.1f} |
| Avg latency (ms) | {S['naive']['avg_latency_ms']:,.0f} | {S['engineered']['avg_latency_ms']:,.0f} |

**Measured token reduction: {saved:.1f}%** ({'meets' if saved > 85 else 'does NOT meet'} the >85% target on this run).
Faithfulness {'maintained or improved' if S['engineered']['faithfulness'] >= S['naive']['faithfulness'] - 1e-9 else 'LOWER than naive on this run'}.
Numbers come from `bench/results.json`; rerun with a real `GROQ_API_KEY` and real embeddings before quoting them.
"""
    (HERE / "results_summary.md").write_text(md); print(md)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4)); lab = ["Faithfulness", "Relevancy", "Ctx precision"]; k = ["faithfulness", "answer_relevancy", "context_precision"]
    x = np.arange(3)
    for i, (m, c) in enumerate((("naive", "#c0392b"), ("engineered", "#1e8e5a"))): ax[0].bar(x + i * .38, [np.nan_to_num(S[m][kk]) for kk in k], .38, label=m, color=c)
    ax[0].set_xticks(x + .19); ax[0].set_xticklabels(lab); ax[0].set_ylim(0, 1.05); ax[0].set_title(f"Quality ({scorer.split(' ')[0]})"); ax[0].legend()
    ax[1].bar(["Naive", "OpsGate"], [S["naive"]["avg_input_tokens"], S["engineered"]["avg_input_tokens"]], color=["#c0392b", "#1e8e5a"]); ax[1].set_title(f"Avg input tokens ({saved:.1f}% saved)")
    plt.tight_layout(); plt.savefig(HERE / "results_summary.png", dpi=150)

if __name__ == "__main__": main()
