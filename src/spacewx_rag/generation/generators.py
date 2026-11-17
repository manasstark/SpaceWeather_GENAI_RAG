"""Two generators behind one interface (docs/architecture.md section 2.5).

- BaselineGenerator: offline and deterministic. Builds the description from card hints and copies
  sentences verbatim from the top retrieved passages. No model, no API key.
- OllamaGenerator: a local LLM through Ollama, constrained to the JSON schema.
"""

from __future__ import annotations

import copy
import os
import re
import time
from dataclasses import asdict, dataclass, field

from pydantic import ValidationError

from spacewx_rag.cards import flatten
from spacewx_rag.generation.prompts import model_card, system_prompt, user_prompt
from spacewx_rag.generation.schema import EventDescription, json_schema
from spacewx_rag.utils.config import load_params


@dataclass
class GenerationResult:
    event_id: str
    generator: str
    model: str
    use_retrieval: bool
    temperature: float
    run: int
    valid: bool
    output: dict | None
    raw: str
    error: str | None
    seconds: float
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    passages: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _passage_record(p: dict) -> dict:
    return {
        k: p.get(k)
        for k in (
            "chunk_id",
            "doc_id",
            "title",
            "section",
            "score",
            "queries",
            "text",
            "source_url",
            "licence_status",
        )
    }


# ---------------------------------------------------------------- offline baseline

PHYSICS_WORDS = {
    "southward",
    "bz",
    "imf",
    "magnetic",
    "reconnection",
    "energy",
    "solar",
    "wind",
    "storm",
    "magnetosphere",
}
AREA_WORDS = {
    "power systems": {"power", "grid", "transformer", "voltage", "currents"},
    "satellites and spacecraft": {
        "satellite",
        "spacecraft",
        "drag",
        "charging",
        "orientation",
        "orbit",
    },
    "GNSS navigation": {"gps", "gnss", "navigation", "positioning"},
    "HF radio": {"hf", "radio", "communication", "propagation"},
    "aurora": {"aurora", "latitude", "seen"},
}


def _sentences(text: str) -> list[str]:
    out = []
    for s in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text)):
        s = s.strip().lstrip(". ")
        if 40 <= len(s) <= 320 and s[0].isupper():
            out.append(s)
    return out


def _best_sentence(text: str, words: set[str]) -> str | None:
    scored = [(sum(w in s.lower() for w in words), s) for s in _sentences(text)]
    scored = [x for x in scored if x[0] > 0]
    return max(scored, key=lambda x: x[0])[1] if scored else None


class BaselineGenerator:
    name = "baseline"
    model = "extractive-v1"

    def generate(
        self, card: dict, passages: list[dict], temperature: float = 0.0, run: int = 0
    ) -> GenerationResult:
        t0 = time.time()
        hints = card["hints"]
        storm = card["type"] == "storm"
        kp_i = next(i for i, h in enumerate(hints) if h.startswith("Kp reached"))
        summary = " ".join([hints[0], hints[kp_i], hints[-1]])
        chars = [{"statement": h, "evidence": [f"hints.{i}"]} for i, h in enumerate(hints)]

        interp, impacts = [], []
        if storm:
            phys = [p for p in passages if "physics" in p.get("queries", [])]
            for p in phys[:2]:
                s = _best_sentence(p["text"], PHYSICS_WORDS)
                if s:
                    interp.append({"statement": s, "sources": [p["chunk_id"]]})
            for area, words in AREA_WORDS.items():
                hits = [p for p in passages if f"impacts: {area}" in p.get("queries", [])]
                for p in hits:
                    s = _best_sentence(p["text"], words)
                    if s:
                        impacts.append({"area": area, "statement": s, "sources": [p["chunk_id"]]})
                        break

        unsupported = [
            "physical cause of the event: no supporting source",
            "comparison with other historical storms: no supporting source",
            "forecast or prediction: not provided",
        ]
        if not impacts:
            unsupported.append(
                "impacts: no supporting source"
                if storm
                else "impacts: none expected for a quiet interval"
            )
        out = {
            "event_id": card["event_id"],
            "summary": summary,
            "characteristics": chars,
            "possible_physical_interpretation": interp,
            "potential_impacts": impacts,
            "caveats": card["data_quality"]["caveats"][3:],
            "unsupported_or_unknown": unsupported,
        }
        EventDescription.model_validate(out)
        return GenerationResult(
            card["event_id"],
            self.name,
            self.model,
            bool(passages),
            temperature,
            run,
            True,
            out,
            "",
            None,
            round(time.time() - t0, 3),
            passages=[_passage_record(p) for p in passages],
        )


# ---------------------------------------------------------------- local LLM


def card_paths(card: dict) -> list[str]:
    """Field paths the model may cite as evidence (everything in the card it is shown)."""
    return sorted(k for k in flatten(model_card(card)) if not k.startswith("claims_requiring"))


def constrained_schema(card: dict, passages: list[dict]) -> dict:
    """The output schema with evidence limited to real card paths and sources limited to the
    retrieved chunk ids, so the decoder cannot invent a citation. With no passages, no sources."""
    s = copy.deepcopy(json_schema())
    d = s["$defs"]
    cap = load_params()["generation"].get("max_list_items")  # guards against repetition loops
    ev = d["Characteristic"]["properties"]["evidence"]
    ev["items"] = {"type": "string", "enum": card_paths(card)}
    if cap:
        ev["maxItems"] = cap
    ids = [p["chunk_id"] for p in passages]
    for name in ("Interpretation", "Impact"):
        prop = d[name]["properties"]["sources"]
        if ids:
            prop["items"] = {"type": "string", "enum": ids}
            if cap:
                prop["maxItems"] = cap
        else:
            prop["maxItems"] = 0
    s["properties"]["event_id"] = {"type": "string", "enum": [card["event_id"]]}
    return s


class OllamaGenerator:
    name = "ollama"

    def __init__(self, model: str | None = None, host: str | None = None):
        import ollama

        g = load_params()["generation"]
        self.model = model or os.environ.get("LLM_MODEL") or g["model"]
        self.client = ollama.Client(host=host or os.environ.get("OLLAMA_HOST") or None)
        self.cfg = g

    def chat_json(
        self,
        system: str | None,
        user: str,
        schema: dict,
        temperature: float,
        num_predict: int | None = None,
        seed_offset: int = 0,
    ) -> tuple[str, dict]:
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": user}
        ]
        r = self.client.chat(
            model=self.model,
            messages=msgs,
            format=schema,
            options={
                "temperature": temperature,
                "num_ctx": self.cfg["num_ctx"],
                "num_predict": num_predict or self.cfg["num_predict"],
                "seed": self.cfg["seed"] + seed_offset,
            },
        )
        return r["message"]["content"], {
            "prompt_tokens": r.get("prompt_eval_count"),
            "output_tokens": r.get("eval_count"),
        }

    def generate(
        self, card: dict, passages: list[dict], temperature: float | None = None, run: int = 0
    ) -> GenerationResult:
        temp = self.cfg["temperature"] if temperature is None else temperature
        t0 = time.time()
        schema = constrained_schema(card, passages)
        raw, meta = self.chat_json(
            system_prompt(), user_prompt(card, passages), schema, temp, seed_offset=run
        )
        out, err = None, None
        try:
            out = EventDescription.model_validate_json(raw).model_dump()
        except ValidationError as e:
            err = f"schema: {e.errors()[0]['msg']} at {e.errors()[0]['loc']}"
        return GenerationResult(
            card["event_id"],
            self.name,
            self.model,
            bool(passages),
            temp,
            run,
            out is not None,
            out,
            raw,
            err,
            round(time.time() - t0, 1),
            meta["prompt_tokens"],
            meta["output_tokens"],
            [_passage_record(p) for p in passages],
        )


def get_generator(provider: str | None = None, model: str | None = None):
    p = provider or os.environ.get("LLM_PROVIDER") or load_params()["generation"]["provider"]
    if p == "baseline":
        return BaselineGenerator()
    if p == "ollama":
        return OllamaGenerator(model)
    raise ValueError(f"unknown LLM provider {p!r} (use 'ollama' or 'baseline')")
