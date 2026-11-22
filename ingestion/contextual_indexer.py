"""Contextual retrieval: embed (summary + chunk), store ONLY the original chunk."""
import hashlib, threading
from functools import lru_cache
from pathlib import Path
import chromadb
from core import config
from core.llm import get_llm, get_embedder, have_llm
from ingestion.markdown_loader import load_document
from ingestion.triage_splitter import chunk_markdown

COLLECTION = "runbooks"

_client_lock = threading.Lock()
_client_obj = None
def _client():
    """One shared Chroma client. Locked: the naive and engineered pipelines now run in parallel threads, and Chroma's first-time
    creation is not thread-safe (opening it per query was also a hidden per-request cost)."""
    global _client_obj
    with _client_lock:
        if _client_obj is None: _client_obj = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        return _client_obj

def get_collection(reset=False, name=COLLECTION):
    client = _client()
    if reset:
        try: client.delete_collection(name)
        except Exception: pass
    return client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})

def situate(doc_md: str, chunk: str, title: str) -> str:
    """1-2 sentence context that places the chunk inside the whole document."""
    if not have_llm():
        return f"From the document '{title}'."
    prompt = (f"<document>\n{doc_md[:6000]}\n</document>\n<chunk>\n{chunk}\n</chunk>\n"
              "Write 1-2 short sentences that situate this chunk within the document to improve search retrieval. Output only those sentences.")
    return get_llm().invoke(prompt).content.strip()

def build_index(reset=True) -> int:
    col, emb, n = get_collection(reset), get_embedder(), 0
    for path in sorted(Path(config.RUNBOOK_DIR).iterdir()):
        if path.suffix.lower() not in (".pdf", ".html", ".md"): continue
        md = load_document(path); title = path.stem.replace("_", " ")
        chunks = chunk_markdown(md)
        enriched = [f"{situate(md, c['text'], title)}\n{c['text']}" for c in chunks]
        vecs = emb.embed(enriched)
        col.add(ids=[hashlib.md5((path.name + c["text"]).encode()).hexdigest() for c in chunks],
                embeddings=vecs, documents=[c["text"] for c in chunks],
                metadatas=[{"source": path.name, "chapter": c["chapter"], "section": c["section"], "kind": c["kind"],
                            "context": e[: e.index("\n")] if "\n" in e else ""} for c, e in zip(chunks, enriched)])
        n += len(chunks)
    return n
