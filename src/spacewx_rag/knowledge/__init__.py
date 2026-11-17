"""Knowledge base (notebook 05): extraction, chunking, embeddings, index."""

from spacewx_rag.knowledge.build import (
    build_index,
    chunk_documents,
    extract,
    extract_all,
    load_chunks,
    load_manifest,
    sections,
)
from spacewx_rag.knowledge.embeddings import embed, get_model

__all__ = [
    "build_index",
    "chunk_documents",
    "embed",
    "extract",
    "extract_all",
    "get_model",
    "load_chunks",
    "load_manifest",
    "sections",
]
