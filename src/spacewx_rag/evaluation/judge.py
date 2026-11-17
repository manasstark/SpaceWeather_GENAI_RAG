"""LLM support judge: does a cited passage support a cause or impact claim? Results are cached
on disk so re-running the evaluation does not repeat model calls.

v1: verdict + reason (prompts/judge_support_v1.md). It proved lenient: it accepted support that
    was not in the passages and G5 effects for a G4 storm (decision 56).
v2 (default): adds a storm-level rule and must return the supporting sentence word for word; a
    "supported" verdict only counts when that quote is found in the cited passages.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from string import Template

from spacewx_rag.generation.prompts import format_passages
from spacewx_rag.utils.config import Paths, load_params

SCHEMAS = {
    "v1": {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["supported", "not_supported"]},
            "reason": {"type": "string"},
        },
        "required": ["verdict", "reason"],
    },
    "v2": {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["supported", "not_supported"]},
            "quote": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["verdict", "quote", "reason"],
    },
}
PROMPTS = {"v1": "judge_support_v1.md", "v2": "judge_support.md"}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def quote_found(quote: str, passages: list[dict], min_chars: int = 25) -> bool:
    """The quote (ignoring case, spacing and punctuation) appears in one of the passages."""
    q = _norm(quote)
    if len(q) < min_chars:
        return False
    return any(q in _norm(p["text"]) for p in passages)


class SupportJudge:
    def __init__(self, generator=None, version: str | None = None, cache_path=None):
        from spacewx_rag.generation.generators import OllamaGenerator

        cfg = load_params()["evaluation"]
        self.version = version or cfg.get("judge_version", "v2")
        self.gen = generator or OllamaGenerator(model=cfg["judge_model"])
        self.temperature = cfg["judge_temperature"]
        self.template = Template(
            (Paths().prompts / PROMPTS[self.version]).read_text(encoding="utf-8")
        )
        self.cache_path = Path(cache_path) if cache_path else Paths().results / "judge_cache.jsonl"
        self.cache = {}
        if self.cache_path.exists():
            with open(self.cache_path, encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    self.cache[d["key"]] = d
        self.calls = 0

    def key(self, claim: str, passages: list[dict]) -> str:
        tag = "" if self.version == "v1" else f"{self.version}|"  # v1 keys predate versioning
        text = tag + self.gen.model + "|" + claim + "|" + ",".join(p["chunk_id"] for p in passages)
        return hashlib.sha256(text.encode()).hexdigest()[:20]

    def __call__(self, claim: str, passages: list[dict]) -> tuple[bool, str, str]:
        """(supported, reason, supporting quote or '')."""
        k = self.key(claim, passages)
        if k not in self.cache:
            prompt = self.template.substitute(
                claim=claim, passages=format_passages(passages)
            ).strip()
            raw, _ = self.gen.chat_json(
                None, prompt, SCHEMAS[self.version], self.temperature, num_predict=300
            )
            quote = ""
            try:
                d = json.loads(raw)
                verdict, reason, quote = (
                    d["verdict"] == "supported",
                    d.get("reason", ""),
                    d.get("quote", ""),
                )
            except (json.JSONDecodeError, KeyError):
                verdict, reason = False, f"unparseable judge output: {raw[:80]}"
            found = quote_found(quote, passages) if self.version != "v1" else None
            self.cache[k] = {
                "key": k,
                "version": self.version,
                "claim": claim,
                "chunks": [p["chunk_id"] for p in passages],
                "verdict_supported": verdict,
                "quote": quote,
                "quote_found": found,
                "supported": verdict if self.version == "v1" else (verdict and bool(found)),
                "reason": reason,
                "judge_model": self.gen.model,
            }
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(self.cache[k], ensure_ascii=False) + "\n")
            self.calls += 1
        d = self.cache[k]
        reason = d["reason"]
        if d.get("verdict_supported") and d.get("quote_found") is False:
            reason = f"[quote not found in passages] {reason}"
        return d["supported"], reason, d.get("quote", "") or ""
