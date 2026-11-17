"""Command-line entry point: `python -m spacewx_rag <stage>` (or the `spacewx` script).

Stages, in order:
  interim    raw DSCOVR NetCDF files -> data/interim/*.parquet       (needs data/raw, slow)
  align      interim + raw indices -> data/processed/aligned_hourly.parquet
  events     aligned table -> data/events/event_table.parquet
  cards      event table -> data/events/event_cards.jsonl (+ evaluation cards)
  kb         raw documents -> texts, chunks.jsonl, Chroma index
  generate   cards + retrieval -> evaluation/results/generations/<run>/
  evaluate   generations -> evaluation/results/*.csv
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from spacewx_rag.utils.config import Paths, load_env


def eval_ids(p: Paths) -> list[str]:
    return pd.read_csv(p.evaluation_set)["event_id"].tolist()


def cmd_interim(a):
    from spacewx_rag.data import build_interim

    build_interim()


def cmd_align(a):
    from spacewx_rag.data import build_aligned

    df = build_aligned(save=True)
    print("aligned:", df.shape)


def cmd_events(a):
    from spacewx_rag.events import build_event_table

    ev = build_event_table(save=True)
    print("events:", ev.shape, ev["cls"].value_counts().to_dict())


def cmd_cards(a):
    from spacewx_rag.cards import build_cards, check_cards, save_cards

    p = Paths()
    ev, al = pd.read_parquet(p.event_table), pd.read_parquet(p.aligned)
    roles = pd.read_csv(p.evaluation_set, index_col="event_id")["role"].to_dict()
    cards = build_cards(ev, al, roles)
    bad = check_cards(cards, ev, al)
    save_cards(cards, list(roles))
    print("cards:", len(cards), "| disagree with aligned table:", bad)


def cmd_kb(a):
    from spacewx_rag.knowledge import build_index, chunk_documents, extract_all

    extract_all()
    chunks = chunk_documents()
    col = build_index(chunks)
    print("chunks:", len(chunks), "| indexed:", col.count())


def cmd_generate(a):
    from spacewx_rag.generation import generate_events, get_generator
    from spacewx_rag.retrieval import Retriever

    p = Paths()
    ids = a.events.split(",") if a.events else eval_ids(p)
    gen = get_generator(a.provider, a.model)
    run_name = a.run or (gen.name if not a.no_retrieval else f"{gen.name}_no_retrieval")
    res = generate_events(
        ids,
        gen,
        None if a.no_retrieval else Retriever(),
        run_name=run_name,
        use_retrieval=not a.no_retrieval,
        temperature=a.temperature,
        runs=a.runs,
        overwrite=a.overwrite,
    )
    print(f"{run_name}: {sum(r['valid'] for r in res)}/{len(res)} valid outputs")


def cmd_evaluate(a):
    from spacewx_rag.evaluation import evaluate_runs

    evaluate_runs(a.runs_to_eval.split(",") if a.runs_to_eval else None, judge=not a.no_judge)


def main(argv=None):
    load_env()
    ap = argparse.ArgumentParser(
        prog="spacewx", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in [
        ("interim", cmd_interim),
        ("align", cmd_align),
        ("events", cmd_events),
        ("cards", cmd_cards),
        ("kb", cmd_kb),
    ]:
        sub.add_parser(name).set_defaults(fn=fn)
    g = sub.add_parser("generate")
    g.add_argument("--events", help="comma-separated ids (default: the 15 evaluation events)")
    g.add_argument("--provider", help="ollama | baseline (default: LLM_PROVIDER or the config)")
    g.add_argument("--model")
    g.add_argument("--run", help="output folder name under evaluation/results/generations/")
    g.add_argument("--no-retrieval", action="store_true", help="ablation: card only, no passages")
    g.add_argument("--temperature", type=float)
    g.add_argument("--runs", type=int, default=1, help="repeated runs per event (consistency)")
    g.add_argument("--overwrite", action="store_true")
    g.set_defaults(fn=cmd_generate)
    e = sub.add_parser("evaluate")
    e.add_argument("--runs-to-eval", help="comma-separated run folders (default: all)")
    e.add_argument("--no-judge", action="store_true", help="skip the LLM support judge")
    e.set_defaults(fn=cmd_evaluate)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
