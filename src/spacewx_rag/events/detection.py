"""Storm and quiet-window detection and per-event features (notebook 03, decisions 22 to 26).

Output: data/events/event_table.parquet (302 events, 33 columns).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from spacewx_rag.utils.config import Paths, load_params

H = pd.Timedelta(hours=1)


def episodes(mask: pd.Series, merge_gap: int) -> list[list[int]]:
    """Runs of True in `mask`; runs separated by fewer than `merge_gap` hours are merged.
    Returns [start_position, end_position_exclusive] pairs."""
    arr = mask.to_numpy().astype(np.int8)
    e = np.diff(np.r_[0, arr, 0])
    starts, ends = np.flatnonzero(e == 1), np.flatnonzero(e == -1)
    merged: list[list[int]] = []
    for s, en in zip(starts, ends):
        if merged and s - merged[-1][1] < merge_gap:
            merged[-1][1] = en
        else:
            merged.append([s, en])
    return merged


def classify(min_dst: pd.Series) -> pd.Series:
    cl = load_params()["events"]["storm"]["classes"]
    edges = [cl["super"][0], cl["super"][1], cl["intense"][1], cl["moderate"][1] + 1]
    return pd.cut(min_dst, edges, labels=["super", "intense", "moderate"])


def detect_storms(aligned: pd.DataFrame) -> pd.DataFrame:
    c = load_params()["events"]["storm"]
    idx, dst = aligned.index, aligned["dst"].to_numpy()
    n = len(dst)
    rows = []
    for s, en in episodes(aligned["dst"] <= c["dst_threshold_nT"], c["merge_gap_hours"]):
        core = dst[s:en]
        k = s + int(np.argmin(core))
        j = en
        while j < n and j < en + c["recovery_cap_hours"] and dst[j] <= c["recovery_level_nT"]:
            j += 1
        rows.append(
            {
                "core_start": idx[s],
                "core_end": idx[en - 1],
                "min_dst": int(core.min()),
                "min_time": idx[k],
                "start": idx[max(s - c["lookback_hours"], 0)],
                "end": idx[min(j, n - 1)],
            }
        )
    ev = pd.DataFrame(rows)
    ev["cls"] = classify(ev["min_dst"]).astype(str)
    ev["type"] = "storm"
    return ev


def quiet_candidates(aligned: pd.DataFrame, storms: pd.DataFrame) -> pd.DataFrame:
    q = load_params()["events"]["quiet"]
    W, idx = q["window_hours"], aligned.index
    fld_dsc = aligned["bz_src"] == "dscovr"
    pla_any = aligned["plasma_src"] != "missing"
    excl = pd.Series(False, index=idx)
    buf = pd.Timedelta(hours=q["buffer_hours_from_storms"])
    for a, b in zip(storms["start"], storms["end"]):
        excl.loc[a - buf : b + buf] = True
    cand = []
    for i in range(0, len(idx) - W + 1, W):
        sl = slice(i, i + W)
        d, k = aligned["dst"].iloc[sl], aligned["kp"].iloc[sl]
        if (
            d.min() > q["dst_above_nT"]
            and k.max() <= q["kp_max"]
            and fld_dsc.iloc[sl].mean() >= q["min_field_dscovr_share"]
            and pla_any.iloc[sl].mean() >= q["min_plasma_any_share"]
            and not excl.iloc[sl].any()
        ):
            cand.append(
                {
                    "start": idx[i],
                    "end": idx[i + W - 1],
                    "min_dst": int(d.min()),
                    "f107": round(aligned["f107_adj"].iloc[sl].mean()),
                }
            )
    return pd.DataFrame(cand)


def sample_quiet(cand: pd.DataFrame, aligned: pd.DataFrame) -> pd.DataFrame:
    """Balanced across solar activity bands (decision 24)."""
    sm = load_params()["events"]["quiet"]["sample"]
    q = cand.copy()
    q["band"] = pd.cut(q["f107"], [0, 100, 150, 1000], labels=["<100", "100-150", ">150"])
    qs = (
        pd.concat(
            [
                q[q["band"] == b].sample(n=n, random_state=sm["random_seed"])
                for b, n in sm["f107_bands"].items()
            ]
        )
        .sort_values("start")
        .reset_index(drop=True)
    )
    qs["type"], qs["cls"] = "quiet", "quiet"
    qs["min_time"] = [aligned["dst"].loc[a:b].idxmin() for a, b in zip(qs["start"], qs["end"])]
    qs["core_start"], qs["core_end"] = pd.NaT, pd.NaT
    return qs


def combine(storms: pd.DataFrame, quiet: pd.DataFrame) -> pd.DataFrame:
    keep = ["type", "cls", "start", "end", "min_dst", "min_time", "core_start", "core_end"]
    ev = pd.concat([storms[keep], quiet[keep]]).sort_values("start").reset_index(drop=True)
    ev["event_id"] = [f"E{i + 1:03d}" for i in range(len(ev))]
    ev["overlaps_next"] = ev["start"].shift(-1) <= ev["end"]
    ev["overlaps_prev"] = ev["overlaps_next"].shift(1, fill_value=False)
    return ev


def features(events: pd.DataFrame, aligned: pd.DataFrame) -> pd.DataFrame:
    """Timing, drivers, indices and data quality over each event window."""
    a = aligned.copy()
    a["vbs"] = a["v"] * (-a["bz"]).clip(lower=0) / 1000  # mV/m
    a["pdyn"] = 1.67e-6 * a["n"] * a["v"] ** 2  # nPa, protons only
    look = load_params()["events"]["storm"]["driver_lookback_hours"] * H
    rows = []
    for r in events.itertuples():
        A = a.loc[r.start : r.end]
        D = a.loc[max(r.min_time - look, r.start) : r.min_time]
        rows.append(
            {
                "event_id": r.event_id,
                "hours_start_to_min": (r.min_time - r.start) / H,
                "recovery_hours": (r.end - r.min_time) / H,
                "core_hours": (r.core_end - r.core_start) / H + 1 if r.type == "storm" else np.nan,
                "bz_min": A["bz"].min(),
                "bz_min_time": A["bz"].idxmin(),
                "hours_bz_le_-5": int((A["bz"] <= -5).sum()),
                "vbs_max": A["vbs"].max(),
                "vbs_mean_12h_before_min": D["vbs"].mean(),
                "v_max": A["v"].max(),
                "n_max": A["n"].max(),
                "pdyn_max": A["pdyn"].max(),
                "bt_max": A["bt"].max(),
                "kp_max": A["kp"].max(),
                "ap_max": A["ap"].max(),
                "f107_mean": A["f107_adj"].mean(),
                "f107_spike_in_window": bool(A["f107_spike"].any()),
                "field_dscovr_pct": (A["bz_src"] == "dscovr").mean() * 100,
                "plasma_dscovr_pct": (A["plasma_src"] == "dscovr").mean() * 100,
                "plasma_missing_pct": (A["plasma_src"] == "missing").mean() * 100,
                "dst_min_is_final": bool(a["dst_version"].loc[r.min_time] == 20),
                "kp_preliminary": bool((A["kp_D"] < 2).any()),
            }
        )
    return pd.DataFrame(rows)


def build_event_table(
    aligned: pd.DataFrame | None = None, paths: Paths | None = None, save: bool = False
) -> pd.DataFrame:
    p = paths or Paths()
    al = pd.read_parquet(p.aligned) if aligned is None else aligned
    storms = detect_storms(al)
    quiet = sample_quiet(quiet_candidates(al, storms), al)
    ev = combine(storms, quiet)
    table = ev.merge(features(ev, al), on="event_id")
    m = load_params()["events"]["storm"]["marginal"]
    table["marginal"] = (
        (table["type"] == "storm")
        & (table["core_hours"] <= m["max_core_hours"])
        & (table["min_dst"] > m["min_dst_above_nT"])
    ).to_numpy()
    if save:
        p.events_dir.mkdir(parents=True, exist_ok=True)
        table.to_parquet(p.event_table)
        table.to_csv(p.events_dir / "event_table.csv", index=False)
    return table
