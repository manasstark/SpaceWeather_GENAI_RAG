"""Real-time snapshot -> hourly table -> event card, with the same feature and hint rules as the
historical pipeline. The window is the last 24 hours that have both solar wind and Dst."""

from __future__ import annotations

import numpy as np
import pandas as pd

from spacewx_rag.cards.builder import make_card, make_caveats
from spacewx_rag.events.detection import classify, features
from spacewx_rag.utils.config import load_params

H = pd.Timedelta(hours=1)


def _frame(rows: list[dict], cols: dict[str, str]) -> pd.DataFrame:
    d = pd.DataFrame(rows)
    if d.empty:
        return pd.DataFrame(columns=list(cols.values()))
    d["time"] = pd.to_datetime(d["time_tag"]).dt.tz_localize(None)
    d = d.drop_duplicates("time").set_index("time").sort_index()
    return d[list(cols)].apply(pd.to_numeric, errors="coerce").rename(columns=cols)


def hourly_table(snap: dict) -> pd.DataFrame:
    """Hourly table with the columns the feature code expects. Solar wind is shifted from L1 to
    Earth arrival (1.5e6 km / speed), as in the historical alignment."""
    cfg, al = load_params()["live"], load_params()["alignment"]
    wind = _frame(
        snap["wind"], {"proton_speed": "v", "proton_density": "n", "proton_temperature": "t"}
    )
    mag = _frame(snap["mag"], {"bz_gsm": "bz", "bt": "bt", "bx_gsm": "bx", "by_gsm": "by"})
    speed = wind["v"].reindex(wind.index.union(mag.index)).interpolate(limit=30).ffill().bfill()
    speed = speed.fillna(al["speed_when_missing_km_s"])

    def arrive(df):
        s = df.copy()
        s.index = s.index + pd.to_timedelta(
            float(al["delay_distance_km"]) / speed.reindex(s.index).to_numpy(), unit="s"
        )
        g = s.groupby(s.index.floor("h"))
        out = g.mean()
        return out[g.count().max(axis=1) >= cfg["min_valid_minutes"]]

    w, m = arrive(wind.dropna(subset=["v"])), arrive(mag.dropna(subset=["bz"]))
    dst = pd.DataFrame(snap["dst"])
    dst = pd.Series(
        pd.to_numeric(dst["dst"]).to_numpy(), index=pd.to_datetime(dst["time_tag"]).dt.floor("h")
    )
    kp = pd.DataFrame(snap["kp"])
    kp_t = pd.to_datetime(kp["time_tag"])
    lo = min(x.index.min() for x in (w, m) if len(x)) if len(w) or len(m) else dst.index.min()
    idx = pd.date_range(
        min(lo, dst.index.min()),
        max(dst.index.max(), *(x.index.max() for x in (w, m) if len(x))),
        freq="h",
    )

    a = pd.DataFrame(index=idx)
    a["dst"] = dst.reindex(idx)
    a["dst_version"] = 0  # real-time quick-look, never final
    slot = pd.Series(pd.to_numeric(kp["Kp"]).to_numpy(), index=kp_t).reindex(
        idx, method="ffill", limit=2
    )
    a["kp"] = slot
    a["ap"] = pd.Series(pd.to_numeric(kp["a_running"]).to_numpy(), index=kp_t).reindex(
        idx, method="ffill", limit=2
    )
    a["kp_D"] = 0  # preliminary
    f = pd.DataFrame(snap["f107"])
    f["day"] = pd.to_datetime(f["time_tag"]).dt.normalize()
    noon = f[f["reporting_schedule"] == "Noon"].groupby("day")["flux"].mean()
    daily = noon.combine_first(f.groupby("day")["flux"].mean())
    a["f107_adj"] = daily.reindex(idx.normalize(), method="ffill").to_numpy()
    a["f107_spike"] = False
    for c in ("bz", "bt", "bx", "by"):
        a[c] = m[c].reindex(idx) if c in m else np.nan
    for c in ("v", "n", "t"):
        a[c] = w[c].reindex(idx) if c in w else np.nan
    # the feature code counts "dscovr" hours; here it means "covered by the active L1 monitor"
    a["bz_src"] = np.where(a["bz"].notna(), "dscovr", "missing")
    a["plasma_src"] = np.where(a["v"].notna(), "dscovr", "missing")
    return a


def live_window(a: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    hours = load_params()["live"]["window_hours"]
    both = a.index[a["dst"].notna() & (a["bz"].notna() | a["v"].notna())]
    end = both.max() if len(both) else a["dst"].dropna().index.max()
    return end - (hours - 1) * H, end


def build_live_card(snap: dict) -> tuple[dict, pd.DataFrame]:
    """(card, hourly table). The card follows the historical format plus live caveats."""
    st = load_params()["events"]["storm"]
    a = hourly_table(snap)
    start, end = live_window(a)
    w = a.loc[start:end]
    dmin = int(w["dst"].min())
    min_time = w["dst"].idxmin()
    if dmin <= st["dst_threshold_nT"]:
        typ, cls = "storm", str(classify(pd.Series([dmin])).iloc[0])
        core = w.index[w["dst"] <= st["dst_threshold_nT"]]
        core_start, core_end = core.min(), core.max()
    elif (
        dmin > load_params()["events"]["quiet"]["dst_above_nT"]
        and w["kp"].max() <= load_params()["events"]["quiet"]["kp_max"]
    ):
        typ, cls, core_start, core_end = "quiet", "quiet", pd.NaT, pd.NaT
    else:
        typ, cls, core_start, core_end = "active", "active", pd.NaT, pd.NaT
    ev = pd.DataFrame(
        [
            {
                "event_id": "LIVE",
                "type": typ,
                "cls": cls,
                "start": start,
                "end": end,
                "min_dst": dmin,
                "min_time": min_time,
                "core_start": core_start,
                "core_end": core_end,
            }
        ]
    )
    r = {**ev.iloc[0].to_dict(), **features(ev, a).iloc[0].to_dict()}
    last_dst = w["dst"].dropna().iloc[-1]
    r.update(
        {
            "recovered": bool(last_dst > st["recovery_level_nT"]),
            "overlaps_prev": False,
            "overlaps_next": False,
            "dst_window_min": dmin,
            "marginal": False,
        }
    )
    card = make_card(r)
    card["event_id"] = f"LIVE-{end:%Y-%m-%dT%H}Z"
    card.pop("evaluation_role", None)

    src = sorted({x.get("source") for x in snap["wind"] + snap["mag"] if x.get("source")})
    fld, pla = round(float(r["field_dscovr_pct"])), round(float(r["plasma_dscovr_pct"]))
    cav = make_caveats(r)[:3] + [
        f"Real-time data from NOAA SWPC fetched {snap['fetched_at']}; not validated and may be revised.",
        f"Solar wind comes from NOAA's active L1 monitor ({', '.join(src) or 'unknown'}), not DSCOVR.",
        "Dst is Kyoto's real-time (quick-look) value.",
        "Kp values are preliminary.",
        "F10.7 is the observed daily flux, not adjusted to 1 AU.",
        "The window ends at the latest available data; conditions may still be changing.",
    ]
    if fld < 90 or pla < 90:
        cav.append(
            f"Solar wind covers {fld}% of the hours for the field and {pla}% for the plasma."
        )
    if typ == "storm" and core_start == start:
        cav.append("The storm was already under way at the start of this window.")
    if typ == "storm" and not r["recovered"]:
        cav.append("Dst had not recovered above -30 nT by the latest data.")
    card["data_quality"] = {
        "solar_wind_monitor": ", ".join(src),
        "field_coverage_pct": fld,
        "plasma_coverage_pct": pla,
        "caveats": cav,
    }
    return card, a
