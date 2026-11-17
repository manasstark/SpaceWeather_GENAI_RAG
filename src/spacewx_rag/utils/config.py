"""Project paths and the parameter file (configs/parameters.yaml)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml


def project_root() -> Path:
    """The folder that holds pyproject.toml. SPACEWX_ROOT overrides it."""
    env = os.environ.get("SPACEWX_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "pyproject.toml").exists():
            return p
    return Path.cwd()


class Paths:
    """Every input and output location used by the pipeline."""

    def __init__(self, root: Path | None = None):
        r = Path(root) if root else project_root()
        self.root = r
        self.raw = r / "data" / "raw"
        self.interim = r / "data" / "interim"
        self.processed = r / "data" / "processed"
        self.aligned = self.processed / "aligned_hourly.parquet"
        self.events_dir = r / "data" / "events"
        self.event_table = self.events_dir / "event_table.parquet"
        self.event_cards = self.events_dir / "event_cards.jsonl"
        self.kb = r / "knowledge_base"
        self.raw_docs = self.kb / "raw_docs"
        self.manifest = self.raw_docs / "MANIFEST.csv"
        self.kb_texts = self.kb / "processed" / "texts"
        self.chunks = self.kb / "processed" / "chunks.jsonl"
        self.vector_store = self.kb / "vector_store"
        self.prompts = r / "prompts"
        self.eval_dir = r / "evaluation"
        self.reference_events = self.eval_dir / "reference_events"
        self.evaluation_set = self.reference_events / "evaluation_set.csv"
        self.reference_points = self.reference_events / "reference_points.json"
        self.retrieval_questions = self.eval_dir / "annotations" / "retrieval_questions.json"
        self.results = self.eval_dir / "results"
        self.generations = self.results / "generations"
        self.config = r / "configs" / "parameters.yaml"


@lru_cache(maxsize=4)
def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_env() -> None:
    """Load the project's .env (LLM_PROVIDER, LLM_MODEL, OLLAMA_HOST) without overriding the shell."""
    from dotenv import load_dotenv

    load_dotenv(project_root() / ".env", override=False)


def load_params(path: str | Path | None = None) -> dict:
    """Read configs/parameters.yaml (cached). Returns a plain dict."""
    return _load(str(path or Paths().config))
