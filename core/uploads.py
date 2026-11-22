"""Files the user attaches in the chat (PDF, txt, md, html, csv, json, log, docx).
Each file is chunked with the same triage splitter as the runbooks and stored in its own Chroma collection ("uploads"),
tagged with the thread key, so a question only ever sees the files attached to ITS thread.
Uploaded text is never run through the cleaner: dates and numbers in a user's document must stay byte-exact."""
import hashlib, re, shutil
from pathlib import Path
from core import config
from core.llm import get_embedder
from core.rag_retriever import _embed_query
from ingestion.contextual_indexer import get_collection
from ingestion.markdown_loader import load_document
from ingestion.triage_splitter import chunk_markdown

COLLECTION = "uploads"
OVERVIEW = re.compile(r"\b(summar\w*|overview|tl;?dr|key points|main points|what('s| is) (this|the) (file|document|pdf|doc)|about (this|the) (file|document|pdf|doc))\b", re.I)

def is_overview(q: str) -> bool: return bool(OVERVIEW.search(q))
def _col(): return get_collection(name=COLLECTION)
def _safe(name: str) -> str: return re.sub(r"[^\w.\- ]", "_", Path(name).name)[:120] or "file"
def _dir(key: str) -> Path: return config.UPLOAD_DIR / re.sub(r"[^\w\-]", "_", key)
def _where(key: str, source: str | None = None):
    return {"$and": [{"thread": key}, {"source": source}]} if source else {"thread": key}

def save_and_index(key: str, filename: str, data: bytes) -> dict:
    """Store the raw file, chunk it, embed it, add it to the thread. Raises ValueError with a message safe to show the user."""
    name = _safe(filename); ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in config.UPLOAD_TYPES: raise ValueError(f"{name}: file type .{ext or '?'} is not supported (use {', '.join(config.UPLOAD_TYPES)}).")
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024: raise ValueError(f"{name}: larger than {config.MAX_UPLOAD_MB} MB.")
    d = _dir(key); d.mkdir(parents=True, exist_ok=True); path = d / name; path.write_bytes(data)
    try: md = load_document(path)
    except Exception as e: path.unlink(missing_ok=True); raise ValueError(f"{name}: could not read the file ({e}).")
    if not md.strip(): path.unlink(missing_ok=True); raise ValueError(f"{name}: no readable text found (a scanned PDF needs OCR first).")
    chunks = [c for c in chunk_markdown(md) if c["text"].strip()]
    col = _col(); col.delete(where=_where(key, name))                 # re-attaching the same name replaces it
    title = name.rsplit(".", 1)[0].replace("_", " ")
    emb = get_embedder()
    for start in range(0, len(chunks), 64):                           # batches keep memory flat on big PDFs
        part = chunks[start:start + 64]
        vecs = emb.embed([f"From '{title}'" + (f", section {c['section']}" if c["section"] else "") + f"\n{c['text']}" for c in part])
        col.add(ids=[hashlib.md5(f"{key}|{name}|{start + i}|{c['text']}".encode()).hexdigest() for i, c in enumerate(part)],
                embeddings=vecs, documents=[c["text"] for c in part],
                metadatas=[{"thread": key, "source": name, "idx": start + i, "section": c["section"] or c["chapter"] or "", "chapter": c["chapter"] or "", "kind": c["kind"]} for i, c in enumerate(part)])
    return {"name": name, "chunks": len(chunks), "chars": len(md)}

def list_files(key: str) -> list[dict]:
    if not key: return []
    got = _col().get(where=_where(key), include=["metadatas"]); counts: dict[str, int] = {}
    for m in got["metadatas"]: counts[m["source"]] = counts.get(m["source"], 0) + 1
    return [{"name": n, "chunks": c} for n, c in sorted(counts.items())]

def has_files(key: str | None) -> bool:
    if not key: return False
    try: return bool(_col().get(where=_where(key), limit=1, include=[])["ids"])
    except Exception: return False

def remove_file(key: str, name: str):
    _col().delete(where=_where(key, name)); (_dir(key) / name).unlink(missing_ok=True)

def delete_thread_files(key: str):
    try: _col().delete(where=_where(key))
    except Exception: pass
    shutil.rmtree(_dir(key), ignore_errors=True)

def _row(doc, meta, score): return {"text": doc, "score": score, "section": meta.get("section"), "source": meta.get("source"), "kind": meta.get("kind"), "origin": "upload"}

def retrieve_uploads(query: str, key: str, k: int | None = None, threshold: float | None = None) -> list[dict]:
    k = k or config.UPLOAD_TOP_K; th = config.UPLOAD_SIM_THRESHOLD if threshold is None else threshold
    col = _col(); n = len(col.get(where=_where(key), include=[])["ids"])
    if n == 0: return []
    res = col.query(query_embeddings=[list(_embed_query(query))], n_results=min(k, n), where=_where(key))
    out = [_row(d, m, round(1 - dist, 4)) for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]
    return [r for r in out if r["score"] >= th]

def overview_chunks(key: str, n: int | None = None) -> list[dict]:
    """For 'summarise this file': an even sample across each file, in reading order (similarity search would pick random pieces)."""
    n = n or config.OVERVIEW_CHUNKS; got = _col().get(where=_where(key), include=["documents", "metadatas"])
    by_file: dict[str, list] = {}
    for d, m in zip(got["documents"], got["metadatas"]): by_file.setdefault(m["source"], []).append((m["idx"], d, m))
    out = []
    for rows in by_file.values():
        rows.sort(key=lambda r: r[0]); per = max(1, n // len(by_file)); step = max(1, len(rows) / per)
        picks = [rows[int(i * step)] for i in range(min(per, len(rows)))]
        out += [_row(d, m, None) for _, d, m in picks]
    return out
