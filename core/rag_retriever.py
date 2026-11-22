"""Hybrid retrieval: dense (ChromaDB, contextual-enriched embeddings) + BM25 lexical -> Reciprocal Rank Fusion -> cross-encoder rerank
(FlashRank, local CPU) -> threshold gate (>= SIM_THRESHOLD, default 0.45).

Separation of concerns: embeddings are built from ENRICHED text, but only the ORIGINAL chunk text + metadata is ever returned.
Exact identifiers ("Error 5042") need the lexical leg; paraphrased questions need the dense leg; RRF merges both without score calibration.
If the reranker cannot load (offline / no model files) a deterministic dense + lexical-coverage blend is used and flagged in `scorer`."""
import re
from functools import lru_cache
from core import config
from core.llm import get_embedder, cosine
from ingestion.contextual_indexer import get_collection

STOP = set("the a an of to in on for and or is are was were be do does did how what which who when where why can could would should i me my we our you your it its this that these those with from by at as about into please show tell give need".split())
RRF_K = 60

def _toks(t: str) -> list[str]: return [w for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in STOP]

@lru_cache(maxsize=256)
def _embed_query(query: str) -> tuple:  # repeated / retried questions skip the embedding model
    return tuple(get_embedder().embed([query])[0])

_corpus: dict = {}
def reset_cache(): _corpus.clear()
def _load_corpus(col):
    """BM25 index over the ORIGINAL chunk texts, rebuilt only when the collection size changes."""
    n = col.count()
    if _corpus.get("n") != n or _corpus.get("name") != col.name:
        from rank_bm25 import BM25Okapi
        got = col.get(include=["documents", "metadatas"])
        toks = [_toks(d) or ["_"] for d in got["documents"]]
        bm = BM25Okapi(toks)
        _corpus.clear(); _corpus.update(n=n, name=col.name, ids=got["ids"], docs=got["documents"], metas=got["metadatas"], toks=[set(t) for t in toks], bm25=bm)
    return _corpus

_ranker = {"obj": None, "failed": False}
def _get_ranker():
    if config.RERANKER != "flashrank" or _ranker["failed"]: return None
    if _ranker["obj"] is None:
        try:
            from flashrank import Ranker
            _ranker["obj"] = Ranker(model_name=config.RERANK_MODEL, cache_dir=str(config.DATA / "flashrank"))
        except Exception as e:
            _ranker["failed"] = True; print(f"[rag] FlashRank unavailable ({type(e).__name__}); using dense+lexical blend")
    return _ranker["obj"]

def _coverage(qt: list[str], doc_toks: set, c: dict) -> float:
    """idf-weighted share of query terms found in the chunk (absolute 0..1, unlike a max-normalised BM25 score)."""
    if not qt: return 0.0
    idf = c["bm25"].idf; top = max(idf.values(), default=1.0) or 1.0
    w = {t: max(idf.get(t, top), 0.1) for t in set(qt)}
    return sum(w[t] for t in w if t in doc_toks) / sum(w.values())

def search(query: str, k: int = 5, hybrid: bool = True, col=None) -> list[dict]:
    """All candidates with final scores (NOT threshold-gated), best first. hybrid=False is the naive dense-only baseline."""
    col = col or get_collection()
    if col.count() == 0: return []
    n = col.count(); q = list(_embed_query(query))
    cand = max(config.HYBRID_CANDIDATES, k)
    res = col.query(query_embeddings=[q], n_results=min(cand, n))
    dense = [{"id": i, "text": d, "meta": m, "sim": 1 - dist} for i, d, m, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])]
    def out(x, score, scorer):
        m = x["meta"]; return {"text": x["text"], "score": round(float(score), 4), "section": m.get("section") or m.get("chapter"), "source": m.get("source"), "kind": m.get("kind"), "scorer": scorer}
    if not hybrid: return [out(x, x["sim"], "dense") for x in dense[:k]]

    c = _load_corpus(col); qt = _toks(query)
    bm = c["bm25"].get_scores(qt) if qt else [0.0] * len(c["ids"])
    lex_rank = [i for i in sorted(range(len(bm)), key=lambda i: -bm[i]) if bm[i] > 0][:cand]
    fused: dict = {}
    for r, x in enumerate(dense): fused[x["id"]] = fused.get(x["id"], 0) + 1 / (RRF_K + r + 1)
    for r, i in enumerate(lex_rank): fused[c["ids"][i]] = fused.get(c["ids"][i], 0) + 1 / (RRF_K + r + 1)
    top = sorted(fused, key=fused.get, reverse=True)[:10]
    idx = {cid: i for i, cid in enumerate(c["ids"])}; sims = {x["id"]: x["sim"] for x in dense}
    missing = [t for t in top if t not in sims]
    if missing:                                                      # BM25-only candidates still need a dense similarity
        emb = col.get(ids=missing, include=["embeddings"])["embeddings"]
        for t, e in zip(missing, emb): sims[t] = cosine(q, e)
    rows = [{"id": t, "text": c["docs"][idx[t]], "meta": c["metas"][idx[t]], "sim": sims[t], "cov": _coverage(qt, c["toks"][idx[t]], c)} for t in top]

    ranker = _get_ranker()
    if ranker:
        from flashrank import RerankRequest
        rr = ranker.rerank(RerankRequest(query=query, passages=[{"id": r["id"], "text": r["text"]} for r in rows]))
        sc = {p["id"]: float(p["score"]) for p in rr}
        return sorted([out(r, sc.get(r["id"], 0.0), "flashrank") for r in rows], key=lambda d: -d["score"])[:k]
    return sorted([out(r, max(r["sim"], 0.4 * r["sim"] + 0.6 * r["cov"]), "hybrid-blend") for r in rows], key=lambda d: -d["score"])[:k]

def retrieve(query: str, k: int = 5, threshold: float | None = None, use_threshold: bool = True, hybrid: bool = True) -> list[dict]:
    th = config.SIM_THRESHOLD if threshold is None else threshold
    return [d for d in search(query, k, hybrid) if not use_threshold or d["score"] >= th]
