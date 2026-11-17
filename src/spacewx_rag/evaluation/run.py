"""Score every generation run and save the tables in evaluation/results/."""

from __future__ import annotations

import pandas as pd

from spacewx_rag.evaluation.metrics import consistency, retrieval_relevance, score_results
from spacewx_rag.generation.pipeline import load_results
from spacewx_rag.utils.config import Paths


def evaluate_runs(
    run_names: list[str] | None = None, judge: bool = True, log=print
) -> dict[str, pd.DataFrame]:
    p = Paths()
    names = run_names or sorted(
        d.name for d in p.generations.iterdir() if d.is_dir() and d.name != "dev"
    )
    j = None
    if judge:
        from spacewx_rag.evaluation.judge import SupportJudge

        j = SupportJudge()
    per_output, claims, cons = [], [], []
    for name in names:
        res = load_results(name)
        for r in res:
            r["run_name"] = name
        po, cl = score_results(res, j)
        per_output.append(po)
        claims.append(cl)
        if res and max(r["run"] for r in res) > 0:
            c = consistency(res, po)
            c.insert(0, "run", name)
            cons.append(c)
        log(
            f"{name}: {len(res)} outputs scored"
            + (f" ({j.calls} new judge calls so far)" if j else "")
        )
    po, cl = pd.concat(per_output, ignore_index=True), pd.concat(claims, ignore_index=True)
    p.results.mkdir(parents=True, exist_ok=True)
    po.to_csv(p.results / "generation_metrics.csv", index=False)
    cl.to_csv(p.results / "claims.csv", index=False)
    summary = run_summary(po)
    summary.to_csv(p.results / "generation_summary.csv")
    out = {"per_output": po, "claims": cl, "summary": summary}
    if cons:
        out["consistency"] = pd.concat(cons, ignore_index=True)
        out["consistency"].to_csv(p.results / "consistency.csv", index=False)

    from spacewx_rag.retrieval import Retriever

    rq, rs = retrieval_relevance(Retriever())
    rq.assign(relevant=rq["relevant"].astype(str), retrieved=rq["retrieved"].astype(str)).to_csv(
        p.results / "retrieval_questions_extended.csv", index=False
    )
    rs.to_csv(p.results / "retrieval_summary.csv")
    out["retrieval"], out["retrieval_summary"] = rq, rs
    log(summary.to_string())
    log(rs.to_string())
    return out


def run_summary(po: pd.DataFrame) -> pd.DataFrame:
    """One row per run (repeat 0 only, so every run is compared on the same 15 events), plus by class."""
    first = po[po["repeat"] == 0]
    cols = {
        "valid": "mean",
        "groundedness": "mean",
        "unsupported_rate": "mean",
        "card_claims_grounded": "mean",
        "source_claims_grounded": "mean",
        "completeness": "mean",
        "must_not_violations": "sum",
        "number_errors": "sum",
        "fabricated_sources": "sum",
        "self_contradiction": "mean",
        "n_claims": "mean",
        "seconds": "mean",
    }
    have = {k: v for k, v in cols.items() if k in first}
    s = first.groupby("run").agg(have)
    s.insert(0, "outputs", first.groupby("run").size())
    return s.round(3)
