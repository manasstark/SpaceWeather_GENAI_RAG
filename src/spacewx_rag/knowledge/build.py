"""Knowledge base: text extraction, chunking, embeddings and the Chroma index (notebook 05)."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from spacewx_rag.utils.config import Paths, load_params

NOAA_ARTICLE_DOCS = (2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19)
WIKI_DROP = (
    "table",
    ".reflist",
    ".navbox",
    "sup.reference",
    ".mw-editsection",
    ".hatnote",
    ".thumb",
    ".metadata",
    ".sistersitebox",
    "ol.references",
)


def tidy(s: str) -> str:
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    return re.sub(r"\n\s*\n+", "\n\n", s).strip()


def html_text(path: Path, selector: str, drop: tuple = ()) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    root = soup.select_one(selector) or soup.body
    for t in root(["script", "style", "nav", "footer", "form", "noscript"]):
        t.decompose()
    for sel in drop:
        for t in root.select(sel):
            t.decompose()
    blocks = []
    for el in root.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        txt = re.sub(r"\s+", " ", el.get_text(" ", strip=True))
        if txt:
            blocks.append(("## " + txt) if el.name in ("h1", "h2", "h3", "h4") else txt)
    joined = "\n\n".join(blocks)
    full = root.get_text("\n", strip=True)
    return tidy(joined if len(joined) >= 0.6 * len(full) else full)


def pdf_text(path: Path) -> str:
    from pypdf import PdfReader

    return tidy("\n\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages))


def extract(doc_id: int, path: Path) -> str:
    """Plain text for one document; the rule depends on where it came from."""
    if doc_id in NOAA_ARTICLE_DOCS:  # NOAA pages: the <article> tag
        return html_text(path, "article")
    if doc_id == 9:  # GFZ
        return html_text(path, "main")
    if doc_id == 13:  # Wikipedia
        t = html_text(path, "#mw-content-text", WIKI_DROP)
        cut = [m.start() for m in re.finditer(r"## (References|External links|See also)", t)]
        t = t[: min(cut)].strip() if cut else t
        return tidy(
            "\n".join(line for line in t.split("\n") if "GLAT" not in line and "MLAT" not in line)
        )
    if doc_id in (11, 20):  # NASA, Kyoto Dst
        return html_text(path, "body")
    if doc_id == 12:  # the paper: stop before the ORCID list
        t = pdf_text(path)
        cut = [m.start() for m in re.finditer(r"\n\s*ORCID iDs\s*\n", t)]
        if cut:
            t = t[: cut[0]]
        return tidy(
            re.sub(r"(?m)^The Astrophysical Journal, 979:49.*$|^Hayakawa et al\.\s*$", "", t)
        )
    return pdf_text(path)  # 1 and 21


def load_manifest(paths: Paths | None = None) -> pd.DataFrame:
    return pd.read_csv((paths or Paths()).manifest)


def extract_all(paths: Paths | None = None) -> pd.DataFrame:
    """Write knowledge_base/processed/texts/NN.txt for every indexed document."""
    p = paths or Paths()
    p.kb_texts.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in load_manifest(p).query("index_by_default").itertuples():
        text = extract(r.doc_id, p.raw_docs / r.file)
        (p.kb_texts / f"{r.doc_id:02d}.txt").write_text(text, encoding="utf-8")
        rows.append({"doc_id": r.doc_id, "role": r.role, "chars": len(text)})
    return pd.DataFrame(rows)


def sections(text: str) -> list[tuple[str, str]]:
    """Split a document into (heading, body) pairs using the '## ' lines."""
    out, head, buf = [], "", []
    for line in text.split("\n"):
        if line.startswith("## "):
            if "".join(buf).strip():
                out.append((head, "\n".join(buf).strip()))
            head, buf = line[3:].strip(), []
        else:
            buf.append(line)
    if "".join(buf).strip():
        out.append((head, "\n".join(buf).strip()))
    return out


def chunk_documents(paths: Paths | None = None) -> pd.DataFrame:
    """Chunks with metadata (decision 38). Saves knowledge_base/processed/chunks.jsonl."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    p = paths or Paths()
    cfg = load_params()["knowledge_base"]["chunking"]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
        separators=["\n\n", "\n", ". ", " "],
    )
    rows = []
    for r in load_manifest(p).query("index_by_default").itertuples():
        text = (p.kb_texts / f"{r.doc_id:02d}.txt").read_text(encoding="utf-8")
        k = 0
        for head, body in sections(text):
            for piece in splitter.split_text(body):
                if len(piece) < cfg["min_chunk_chars"]:  # drop fragments too short to carry meaning
                    continue
                rows.append(
                    {
                        "chunk_id": f"d{r.doc_id:02d}_c{k:03d}",
                        "doc_id": r.doc_id,
                        "role": r.role,
                        "title": r.title,
                        "section": head,
                        "source_url": r.source_url,
                        "licence_status": r.licence_status,
                        "text": piece,
                    }
                )
                k += 1
    chunks = pd.DataFrame(rows)
    chunks.to_json(p.chunks, orient="records", lines=True)
    return chunks


def load_chunks(paths: Paths | None = None) -> pd.DataFrame:
    return pd.read_json((paths or Paths()).chunks, lines=True)


def build_index(chunks: pd.DataFrame, paths: Paths | None = None):
    """Embed the chunks locally and (re)create the Chroma collection. Returns the collection."""
    import chromadb
    from chromadb.config import Settings

    from spacewx_rag.knowledge.embeddings import get_model

    p = paths or Paths()
    cfg = load_params()["knowledge_base"]
    model = get_model()
    texts = [f"{r.title}. {r.text}" for r in chunks.itertuples()]
    emb = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=cfg["embedding"]["normalize"],
        show_progress_bar=False,
    )
    client = chromadb.PersistentClient(
        path=str(p.vector_store), settings=Settings(anonymized_telemetry=False)
    )
    name = cfg["vector_store"]["collection"]
    if name in {getattr(c, "name", c) for c in client.list_collections()}:
        client.delete_collection(name)
    col = client.create_collection(name, metadata={"hnsw:space": cfg["vector_store"]["metric"]})
    col.add(
        ids=chunks["chunk_id"].tolist(),
        embeddings=emb.tolist(),
        documents=chunks["text"].tolist(),
        metadatas=[
            {
                "doc_id": int(r.doc_id),
                "role": r.role,
                "title": r.title,
                "section": r.section,
                "source_url": r.source_url,
                "licence_status": r.licence_status,
            }
            for r in chunks.itertuples()
        ],
    )
    return col
