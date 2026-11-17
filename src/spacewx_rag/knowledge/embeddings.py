"""The local embedding model (decision 35), loaded once per process."""

from __future__ import annotations

from functools import lru_cache

from spacewx_rag.utils.config import load_params


@lru_cache(maxsize=1)
def get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(load_params()["knowledge_base"]["embedding"]["model"], device="cpu")


def embed(texts: list[str], query: bool = False):
    """Normalised embeddings; `query=True` adds the bge query prefix."""
    cfg = load_params()["knowledge_base"]["embedding"]
    if query:
        texts = [cfg["query_prefix"] + t for t in texts]
    return get_model().encode(texts, normalize_embeddings=cfg["normalize"], show_progress_bar=False)
