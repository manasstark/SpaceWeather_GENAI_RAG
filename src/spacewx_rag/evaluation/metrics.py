"""Metrics: groundedness (main), unsupported-claim rate, completeness, must-not violations,
consistency across repeated runs, and retrieval relevance on the labelled question set."""

from __future__ import annotations

import itertools
import json
import re

import numpy as np
import pandas as pd

from spacewx_rag.cards import load_cards
from spacewx_rag.evaluation.claims import check_output, self_contradiction
from spacewx_rag.utils.config import Paths

SECTIONS = [
    "summary",
    "characteristics",
    "possible_physical_interpretation",
    "potential_impacts",
    "caveats",
    "unsupported_or_unknown",
]


def section_text(output: dict, section: str) -> str:
    def one(name):
        v = output.get(name, [])
        if isinstance(v, str):
            return v
        return "\n".join(x["statement"] if isinstance(x, dict) else str(x) for x in v)

    if section == "all":
        return "\n".join(one(s) for s in SECTIONS)
    return "\n".join(one(s) for s in section.split("+"))


def reference_checks(output: dict, ref: dict) -> dict:
    """Completeness (share of must-include points covered) and the list of must-not violations."""

    def hit(pt):
        if pt["type"] == "empty":
            return len(output.get(pt["section"], [])) == 0
        return (
            re.search(pt["pattern"], section_text(output, pt["section"]), re.IGNORECASE) is not None
        )

    inc = {p["id"]: hit(p) for p in ref["must_include"]}
    nots = [p["id"] for p in ref["must_not"] if (not hit(p) if p["type"] == "empty" else hit(p))]
    return {
        "completeness": float(np.mean(list(inc.values()))),
        "missed": [k for k, v in inc.items() if not v],
        "must_not_violations": nots,
    }


def load_reference_points(paths: Paths | None = None) -> dict:
    return json.load(open((paths or Paths()).reference_points, encoding="utf-8"))["events"]


def score_results(results: list[dict], judge=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-output metrics and the per-claim table for a list of generation results."""
    cards, refs = load_cards(), load_reference_points()
    rows, claim_rows = [], []
    for r in results:
        card, eid = cards[r["event_id"]], r["event_id"]
        row = {
            "run": r.get("run_name"),
            "event_id": eid,
            "class": card["class"],
            "repeat": r["run"],
            "generator": r["generator"],
            "model": r["model"],
            "use_retrieval": r["use_retrieval"],
            "valid": r["valid"],
            "seconds": r["seconds"],
            "n_passages": len(r.get("passages", [])),
        }
        if r["valid"]:
            claims = check_output(r["output"], card, r.get("passages", []), judge)
            for c in claims:
                claim_rows.append({"run": row["run"], "event_id": eid, "repeat": r["run"], **c})
            cl = pd.DataFrame(claims)
            refc = reference_checks(r["output"], refs[eid])
            interp_imp = cl[cl["kind"].isin(["interpretation", "impact"])]
            row.update(
                {
                    "n_claims": len(cl),
                    "n_grounded": int(cl["grounded"].sum()),
                    "groundedness": float(cl["grounded"].mean()) if len(cl) else np.nan,
                    "unsupported_rate": float(1 - cl["grounded"].mean()) if len(cl) else np.nan,
                    "n_card_claims": int(cl["kind"].isin(["summary", "characteristic"]).sum()),
                    "card_claims_grounded": float(
                        cl.loc[cl["kind"].isin(["summary", "characteristic"]), "grounded"].mean()
                    ),
                    "n_source_claims": len(interp_imp),
                    "source_claims_grounded": float(interp_imp["grounded"].mean())
                    if len(interp_imp)
                    else np.nan,
                    "number_errors": int((~cl["numbers_ok"]).sum()),
                    "fabricated_sources": int(
                        sum(
                            len(x)
                            for x in cl.get("fabricated_sources", pd.Series(dtype=object)).dropna()
                        )
                    ),
                    "n_impacts": int((cl["kind"] == "impact").sum()),
                    "completeness": refc["completeness"],
                    "missed_points": ";".join(refc["missed"]),
                    "must_not_violations": len(refc["must_not_violations"]),
                    "self_contradiction": self_contradiction(r["output"]),
                    "violations": ";".join(refc["must_not_violations"]),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(claim_rows)


def consistency(results: list[dict], per_output: pd.DataFrame) -> pd.DataFrame:
    """For events generated several times: summary similarity, evidence overlap and metric spread."""
    from spacewx_rag.knowledge.embeddings import embed

    out = []
    by_event: dict[str, list[dict]] = {}
    for r in results:
        if r["valid"]:
            by_event.setdefault(r["event_id"], []).append(r)
    for eid, rs in by_event.items():
        if len(rs) < 2:
            continue
        e = embed([x["output"]["summary"] for x in rs])
        sims = [float(e[i] @ e[j]) for i, j in itertools.combinations(range(len(rs)), 2)]
        ev = [{p for c in x["output"]["characteristics"] for p in c["evidence"]} for x in rs]
        jac = [len(a & b) / max(len(a | b), 1) for a, b in itertools.combinations(ev, 2)]
        src = [
            {
                s
                for k in ("possible_physical_interpretation", "potential_impacts")
                for it in x["output"][k]
                for s in it["sources"]
            }
            for x in rs
        ]
        sjac = [len(a & b) / max(len(a | b), 1) for a, b in itertools.combinations(src, 2) if a | b]
        po = per_output[per_output["event_id"] == eid]
        out.append(
            {
                "event_id": eid,
                "class": po["class"].iloc[0],
                "runs": len(rs),
                "summary_similarity": np.mean(sims),
                "evidence_jaccard": np.mean(jac),
                "source_jaccard": np.mean(sjac) if sjac else np.nan,
                "groundedness_mean": po["groundedness"].mean(),
                "groundedness_range": po["groundedness"].max() - po["groundedness"].min(),
                "completeness_range": po["completeness"].max() - po["completeness"].min(),
                "n_impacts": "/".join(str(int(x)) for x in po["n_impacts"]),
            }
        )
    return pd.DataFrame(out)


def retrieval_relevance(
    retriever, paths: Paths | None = None, k: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """hit@1/3/5 and MRR on the labelled questions, by question set."""
    p = paths or Paths()
    with open(p.retrieval_questions, encoding="utf-8") as f:
        qs = json.load(f)["questions"]
    rows = []
    for q in qs:
        res = retriever.search(
            q["question"], k, retriever.general_filter if q["filter"] == "general" else None
        )
        docs = [r["doc_id"] for r in res]
        rank = next((i + 1 for i, d in enumerate(docs) if d in q["relevant_docs"]), np.nan)
        rows.append(
            {
                "set": q["set"],
                "question": q["question"],
                "relevant": q["relevant_docs"],
                "retrieved": docs,
                "rank": rank,
                "top1_score": res[0]["score"],
            }
        )
    df = pd.DataFrame(rows)

    def agg(d):
        return pd.Series(
            {
                "questions": len(d),
                "hit@1": (d["rank"] == 1).mean(),
                "hit@3": (d["rank"] <= 3).mean(),
                "hit@5": (d["rank"] <= 5).mean(),
                "MRR": (1 / d["rank"]).fillna(0).mean(),
            }
        )

    summary = pd.concat(
        [df.groupby("set").apply(agg, include_groups=False), agg(df).to_frame("all").T]
    )
    return df, summary.round(3)
