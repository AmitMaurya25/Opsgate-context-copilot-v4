"""Markdown-aware loaders: PDF (pymupdf4llm) and HTML (cleaned, then markdownify)."""
from pathlib import Path

def load_pdf(path) -> str:
    import pymupdf4llm
    return pymupdf4llm.to_markdown(str(path))

def load_html(path) -> str:
    from bs4 import BeautifulSoup
    from markdownify import markdownify as md
    soup = BeautifulSoup(Path(path).read_text(encoding="utf-8"), "html.parser")
    for t in soup(["script", "style", "nav", "footer", "header", "noscript"]): t.decompose()
    return md(str(soup.body or soup), heading_style="ATX").strip()

def load_docx(path) -> str:
    import docx                                              # python-docx
    d, out = docx.Document(str(path)), []
    for para in d.paragraphs:
        t = para.text.strip()
        if not t: continue
        name = para.style.name or ""
        if name == "Title": out.append("# " + t)
        elif name.startswith("Heading") and name.split()[-1].isdigit(): out.append("#" * min(3, max(1, int(name.split()[-1]))) + " " + t)
        else: out.append(t)
    for tbl in d.tables:                                     # tables are appended after the text (position in the page is not kept)
        rows = [[c.text.strip().replace("|", "/") for c in r.cells] for r in tbl.rows]
        if rows: out.append("\n".join(["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])] + ["| " + " | ".join(r) + " |" for r in rows[1:]]))
    return "\n\n".join(out)

def load_document(path) -> str:
    p = Path(path)
    if p.suffix.lower() == ".pdf": return load_pdf(p)
    if p.suffix.lower() in (".html", ".htm"): return load_html(p)
    if p.suffix.lower() == ".docx": return load_docx(p)
    return p.read_text(encoding="utf-8", errors="replace")
