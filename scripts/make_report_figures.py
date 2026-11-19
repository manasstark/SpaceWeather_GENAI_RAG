"""Figures for the technical report and slides, made from the saved results (no model calls).

    python scripts/make_report_figures.py [--out reports/figures/report]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "evaluation" / "results"
C = {"rag": "#E07B00", "norag": "#5A6390", "base": "#2A9D8F", "red": "#D62839", "purple": "#7B2CBF",
     "blue": "#1D70B8", "grey": "#8A8FA3"}
RUNS = {"ollama_rag": ("LLM + retrieval", C["rag"]), "ollama_no_retrieval": ("LLM, no retrieval", C["norag"]),
        "baseline": ("Offline baseline", C["base"])}
CLASSES = ["super", "intense", "moderate", "quiet"]

mpl.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9, "axes.titlesize": 9.5,
    "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25,
    "grid.linewidth": .6, "savefig.bbox": "tight", "savefig.dpi": 300, "figure.dpi": 150,
})


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png")
    plt.close(fig)
    print("saved", name)


def storm_e216(out):
    a = pd.read_parquet(ROOT / "data" / "processed" / "aligned_hourly.parquet").loc["2024-05-09 19:00":"2024-05-16 17:00"]
    fig, ax = plt.subplots(4, 1, figsize=(6.6, 5.6), sharex=True, gridspec_kw={"hspace": .12})
    ax[0].plot(a.index, a.dst, color=C["purple"], lw=1.4)
    ax[0].fill_between(a.index, a.dst, 0, color=C["purple"], alpha=.12)
    for y, t in ((-50, "storm"), (-250, "super")):
        ax[0].axhline(y, ls=":", lw=.8, color=C["grey"])
        ax[0].text(a.index[-1], y, f" {t}", va="center", fontsize=7, color=C["grey"])
    ax[0].set_ylabel("Dst (nT)")
    ax[1].plot(a.index, a.bz, color=C["red"], lw=1.1)
    ax[1].axhline(0, lw=.7, color="k")
    ax[1].set_ylabel("$B_z$ GSM (nT)")
    om = a.plasma_src == "omni"
    ax[2].plot(a.index, a.v.where(~om), color=C["blue"], lw=1.2, label="DSCOVR")
    ax[2].plot(a.index, a.v.where(om), color=C["rag"], lw=1.2, ls="--", label="OMNI fallback")
    ax[2].set_ylabel("$v$ (km/s)")
    ax[2].legend(loc="upper right", frameon=False, ncol=2)
    cols = [C["red"] if k >= 7 else C["rag"] if k >= 5 else C["base"] for k in a.kp.fillna(0)]
    ax[3].bar(a.index, a.kp, width=1 / 24, color=cols)
    ax[3].axhline(5, ls=":", lw=.8, color=C["grey"])
    ax[3].set_ylabel("Kp")
    ax[3].set_ylim(0, 9.3)
    for x in ax:
        x.axvspan(pd.Timestamp("2024-05-10 07:00"), pd.Timestamp("2024-05-16 05:00"), color="#FFD166", alpha=.12, lw=0)
    fig.autofmt_xdate()
    save(fig, out, "fig_storm_e216")


def timeshift(out):
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.4), gridspec_kw={"width_ratios": [1.4, 1]})
    labels = ["no shift", "fixed 60 min", "$1.5\\times10^6$ km$/v$"]
    vals = [0.718, 0.909, 0.919]
    b = ax[0].bar(labels, vals, color=[C["grey"], C["norag"], C["rag"]], width=.6)
    ax[0].bar_label(b, fmt="%.3f", fontsize=8, padding=2)
    ax[0].set_ylim(.6, .97)
    ax[0].set_ylabel("corr($B_z$ DSCOVR, $B_z$ OMNI)")
    ax[0].set_title("Hourly $B_z$ agreement with OMNI")
    b2 = ax[1].bar(["fixed 60 min", "distance / speed"], [0.828, 0.600], color=[C["norag"], C["rag"]], width=.55)
    ax[1].bar_label(b2, fmt="%.3f nT", fontsize=8, padding=2)
    ax[1].set_ylim(0, 1.0)
    ax[1].set_title("Fast wind ($>600$ km/s), median $|\\Delta B_z|$")
    save(fig, out, "fig_timeshift")


def sources(out):
    a = pd.read_parquet(ROOT / "data" / "processed" / "aligned_hourly.parquet")
    f = a.bz_src.value_counts(normalize=True) * 100
    p = a.plasma_src.value_counts(normalize=True) * 100
    fig, ax = plt.subplots(figsize=(6.2, 1.35))
    left = np.zeros(2)
    for k, col in (("dscovr", C["blue"]), ("omni", C["rag"]), ("missing", C["grey"])):
        v = np.array([f.get(k, 0), p.get(k, 0)])
        ax.barh(["magnetic field", "plasma"], v, left=left, color=col, label=k.upper() if k != "missing" else "missing")
        for i, x in enumerate(v):
            if x > 4:
                ax.text(left[i] + x / 2, i, f"{x:.1f}%", ha="center", va="center", color="white", fontsize=8)
        left += v
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of the 87,024 hours (%)")
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.45))
    ax.grid(False)
    save(fig, out, "fig_sources")


def sensitivity(out):
    gaps = [1, 6, 12, 24]
    data = {-50: [351, 228, 202, 175], -100: [48, 36, 33, 31], -250: [3, 2, 2, 2]}
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.3))
    for thr, col in zip(data, (C["rag"], C["red"], C["purple"])):
        ax[0].plot(gaps, data[thr], marker="o", ms=3.5, color=col, label=f"Dst $\\leq$ {thr} nT")
    ax[0].set_yscale("log")
    ax[0].set_xticks(gaps)
    ax[0].set_xlabel("merge gap (hours)")
    ax[0].set_ylabel("number of events")
    ax[0].axvline(12, ls=":", color=C["grey"], lw=.8)
    ax[0].legend(frameon=False)
    ax[0].set_title("Event count vs. merge gap")
    ev = pd.read_parquet(ROOT / "data" / "events" / "event_table.parquet")
    counts = ev.cls.value_counts().reindex(["quiet", "moderate", "intense", "super"])
    b = ax[1].bar(counts.index, counts.values, color=[C["base"], C["rag"], C["red"], C["purple"]])
    ax[1].bar_label(b, fontsize=8, padding=2)
    ax[1].set_title("The 302 events")
    ax[1].set_ylim(0, 200)
    save(fig, out, "fig_events")


def groundedness_by_class(out, po):
    f = po[po.repeat == 0]
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
    for j, metric in enumerate(["groundedness", "completeness"]):
        piv = f.pivot_table(index="class", columns="run", values=metric).reindex(CLASSES)
        x = np.arange(len(CLASSES))
        for i, (run, (lab, col)) in enumerate(RUNS.items()):
            ax[j].bar(x + (i - 1) * .26, piv[run], width=.26, color=col, label=lab)
        ax[j].set_xticks(x, CLASSES)
        ax[j].set_ylim(0, 1.08)
        ax[j].set_title(metric.capitalize() + " by event class")
    ax[0].legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(1.1, 1.3))
    save(fig, out, "fig_by_class")


def rag_vs_norag(out, po):
    f = po[(po.repeat == 0) & po.run.isin(["ollama_rag", "ollama_no_retrieval"])]
    m = f.groupby("run")[["groundedness", "card_claims_grounded", "source_claims_grounded", "completeness"]].mean()
    labels = ["groundedness\n(all claims)", "card claims\ngrounded", "cause/impact\nclaims grounded", "completeness"]
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(5.4, 2.4))
    for i, run in enumerate(["ollama_rag", "ollama_no_retrieval"]):
        lab, col = RUNS[run]
        b = ax.bar(x + (i - .5) * .36, m.loc[run].values, width=.36, color=col, label=lab)
        ax.bar_label(b, fmt="%.2f", fontsize=7, padding=1.5)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(.5, 1.18))
    save(fig, out, "fig_rag_vs_norag")


def failure_reasons(out, cl):
    def reason(r):
        if r["numbers_ok"] is False or r["numbers_ok"] == "False":
            return "number not in data"
        if str(r.get("no_impact")) == "False":
            return "impact in summary"
        if str(r.get("no_cause")) == "False":
            return "cause without source"
        if str(r.get("sources_ok")) == "False":
            return "no valid source"
        if str(r.get("level_ok")) == "False":
            return "effect from higher G level"
        if str(r.get("judge_supported")) == "False":
            return "judge: not supported"
        return "other"
    bad = cl[(cl.repeat == 0) & ~cl.grounded.astype(bool) & cl.run.isin(["ollama_rag", "ollama_no_retrieval"])].copy()
    bad["reason"] = bad.apply(reason, axis=1)
    tab = bad.groupby(["reason", "run"]).size().unstack(fill_value=0).reindex(columns=["ollama_rag", "ollama_no_retrieval"], fill_value=0)
    tab = tab.loc[tab.sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(5.4, 2.3))
    y = np.arange(len(tab))
    for i, run in enumerate(tab.columns):
        lab, col = RUNS[run]
        b = ax.barh(y + (i - .5) * .38, tab[run], height=.38, color=col, label=lab)
        ax.bar_label(b, labels=[str(v) if v else "" for v in tab[run]], fontsize=7, padding=2)
    ax.set_yticks(y, tab.index)
    ax.set_xlabel("ungrounded claims (15 events, first run)")
    ax.legend(frameon=False, loc="lower right")
    save(fig, out, "fig_failures")


def judge_audit(out):
    # counts from evaluation/annotations/judge_audit_ollama_rag.csv vs judge verdicts (notebook 07)
    mats = {"judge v1": [[14, 0], [5, 2]], "judge v2 (quote)": [[14, 0], [4, 3]], "v2 + G-level check": [[14, 0], [2, 5]]}
    fig, ax = plt.subplots(1, 3, figsize=(6.6, 2.1), gridspec_kw={"wspace": .08})
    for k, (a, (title, m)) in enumerate(zip(ax, mats.items())):
        m = np.array(m)
        a.imshow(m, cmap="Oranges", vmin=0, vmax=16)
        for i in range(2):
            for j in range(2):
                a.text(j, i, m[i, j], ha="center", va="center", fontsize=11, color="k")
        a.set_xticks([0, 1], ["accepted", "rejected"])
        a.set_yticks([0, 1], ["supported", "not supp."] if k == 0 else ["", ""])
        a.set_title(f"{title}\nagree {m[0, 0] + m[1, 1]}/21", fontsize=8.5)
        a.grid(False)
    ax[0].set_ylabel("manual label")
    fig.text(.5, -.04, "judge verdict", ha="center", fontsize=9)
    save(fig, out, "fig_judge_audit")


def retrieval(out):
    rs = pd.read_csv(RES / "retrieval_summary.csv", index_col=0)
    rows = {"original_18": "original 18 (labels revised)", "extended_2026_11_17": "new 24 (labelled blind)"}
    ms = ["hit@1", "hit@3", "hit@5", "MRR"]
    fig, ax = plt.subplots(figsize=(4.6, 2.2))
    x = np.arange(len(ms))
    for i, (k, lab) in enumerate(rows.items()):
        b = ax.bar(x + (i - .5) * .36, rs.loc[k, ms].values, width=.36, color=[C["norag"], C["rag"]][i], label=lab)
        ax.bar_label(b, fmt="%.2f", fontsize=7, padding=1.5)
    ax.set_xticks(x, ms)
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(.5, 1.2))
    save(fig, out, "fig_retrieval")


def consistency(out):
    c = pd.read_csv(RES / "consistency.csv")
    c["label"] = c.event_id + "\n(" + c["class"] + ")"
    fig, ax = plt.subplots(figsize=(5.6, 2.3))
    x = np.arange(len(c))
    for i, (col, lab, colr) in enumerate([("summary_similarity", "summary similarity (cosine)", C["rag"]),
                                          ("evidence_jaccard", "card fields cited (Jaccard)", C["blue"]),
                                          ("source_jaccard", "passages cited (Jaccard)", C["grey"])]):
        ax.bar(x + (i - 1) * .26, c[col], width=.26, color=colr, label=lab)
    for i, s in enumerate(c.n_impacts):
        ax.text(i, 1.05, f"impacts {s}", ha="center", fontsize=7, color=C["red"])
    ax.set_xticks(x, c.label)
    ax.set_ylim(0, 1.2)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.25), fontsize=7)
    save(fig, out, "fig_consistency")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "reports" / "figures" / "report"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    po = pd.read_csv(RES / "generation_metrics.csv")
    cl = pd.read_csv(RES / "claims.csv")
    storm_e216(out)
    timeshift(out)
    sources(out)
    sensitivity(out)
    groundedness_by_class(out, po)
    rag_vs_norag(out, po)
    failure_reasons(out, cl)
    judge_audit(out)
    retrieval(out)
    consistency(out)


if __name__ == "__main__":
    main()
