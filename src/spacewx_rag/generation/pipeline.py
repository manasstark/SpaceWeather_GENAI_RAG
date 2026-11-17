"""Card -> retrieval -> prompt -> generator, for a list of events. Results are saved one file per
event and run, so a long local-LLM run can stop and resume without repeating finished events."""

from __future__ import annotations

import json
from pathlib import Path

from spacewx_rag.cards import load_cards
from spacewx_rag.utils.config import Paths


def result_path(run_name: str, event_id: str, run: int = 0, paths: Paths | None = None) -> Path:
    p = paths or Paths()
    return (
        p.generations / run_name / (f"{event_id}.json" if run == 0 else f"{event_id}_r{run}.json")
    )


def generate_events(
    event_ids: list[str],
    generator,
    retriever=None,
    run_name: str = "rag",
    use_retrieval: bool = True,
    temperature: float | None = None,
    runs: int = 1,
    overwrite: bool = False,
    log=print,
) -> list[dict]:
    """Generate descriptions for `event_ids`, `runs` times each. Returns the result dicts."""
    cards = load_cards()
    out = []
    for eid in event_ids:
        card = cards[eid]
        passages = retriever.passages_for_event(card) if (use_retrieval and retriever) else []
        for run in range(runs):
            path = result_path(run_name, eid, run)
            if path.exists() and not overwrite:
                out.append(json.loads(path.read_text()))
                continue
            kw = {} if temperature is None else {"temperature": temperature}
            res = generator.generate(card, passages, run=run, **kw).to_dict()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(res, indent=1, ensure_ascii=False))
            log(
                f"{run_name} {eid} run {run}: valid={res['valid']} {res['seconds']}s "
                f"tokens in/out={res.get('prompt_tokens')}/{res.get('output_tokens')}"
            )
            out.append(res)
    return out


def load_results(run_name: str, paths: Paths | None = None) -> list[dict]:
    d = (paths or Paths()).generations / run_name
    return [json.loads(f.read_text()) for f in sorted(d.glob("E*.json"))]
