"""3-way triage: tag each markdown block prose/code/table, then route to a specialised splitter."""
import re
from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

@dataclass
class Element:
    kind: str          # prose | code | table
    text: str
    chapter: str = ""
    section: str = ""

PYISH = re.compile(r"^\s*(def |class |import |from \w+ import|for |while |if |try:|except|return |@\w+)")

def partition(md: str) -> list[Element]:
    els, chapter, section = [], "", ""
    lines = md.splitlines(); i = 0; buf = []
    def flush():
        nonlocal buf
        txt = "\n".join(buf).strip()
        if txt: els.append(Element("prose", txt, chapter, section))
        buf = []
    while i < len(lines):
        ln = lines[i]; s = ln.strip()
        if s.startswith("#"):
            flush(); title = s.lstrip("#").strip().strip("*")
            if s.startswith("# "): chapter, section = title, ""
            else: section = title
            i += 1; continue
        if s.startswith("```"):                       # fenced code
            flush(); j = i + 1; blk = []
            while j < len(lines) and not lines[j].strip().startswith("```"): blk.append(lines[j]); j += 1
            els.append(Element("code", "\n".join(blk), chapter, section)); i = j + 1; continue
        if s.startswith("|"):                         # markdown table
            flush(); blk = []
            while i < len(lines) and lines[i].strip().startswith("|"): blk.append(lines[i]); i += 1
            els.append(Element("table", "\n".join(blk), chapter, section)); continue
        if PYISH.match(ln) or (buf and ln.startswith(("    ", "\t")) and any(PYISH.match(b) for b in buf[-6:])):
            # indented / python-looking run -> code
            flush(); blk = []
            while i < len(lines) and (lines[i].strip() == "" or lines[i].startswith((" ", "\t")) or PYISH.match(lines[i])):
                blk.append(lines[i]); i += 1
            els.append(Element("code", "\n".join(blk).rstrip(), chapter, section)); continue
        buf.append(ln); i += 1
    flush()
    return els

_prose = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=60)
_code = RecursiveCharacterTextSplitter.from_language(Language.PYTHON, chunk_size=1500, chunk_overlap=0)

def route_and_split(el: Element) -> list[str]:
    if el.kind == "prose": return _prose.split_text(el.text)
    if el.kind == "code": return _code.split_text(el.text)
    return [el.text]                                   # tables stay whole

def chunk_markdown(md: str) -> list[dict]:
    out = []
    for el in partition(md):
        for c in route_and_split(el):
            out.append({"text": c, "kind": el.kind, "chapter": el.chapter, "section": el.section})
    return out
