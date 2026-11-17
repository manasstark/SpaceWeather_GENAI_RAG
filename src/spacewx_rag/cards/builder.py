"""Event cards: facts, rule-based hints and caveats for one event (from notebook 04).

A card never says what caused an event or what it affected; that can only come from
retrieved documents (decision 29).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from spacewx_rag.utils.config import Paths, load_params

CLAIMS_NEED_SOURCE = [
    "physical cause of the event (for example a coronal mass ejection or a high-speed stream)",
    "impacts on satellites, GNSS, radio communication, power grids or aurora",
    "comparison with, or ranking against, other historical storms",
    "any forecast or prediction",
]


def _hp() -> dict:
    return load_params()["cards"]


def hrs(n) -> str:
    return f"{int(n)} hour" + ("" if int(n) == 1 else "s")


def iso(t) -> str | None:
    return None if pd.isna(t) else pd.Timestamp(t).strftime("%Y-%m-%dT%H:%MZ")


def g_level(kp: float) -> str | None:
    """Approximate NOAA G-scale from the maximum Kp (>= 5, 6, 7, 8, 9)."""
    th = _hp()["hints"]["g_level_kp_thresholds"]
    for name, lim in sorted(th.items(), key=lambda x: -x[1]):
        if kp >= lim - 0.01:
            return name
    return None


def solar_level(f107: float) -> str:
    lv = _hp()["hints"]["f107_levels"]
    return "low" if f107 < lv["low_below"] else ("moderate" if f107 <= lv["high_above"] else "high")


def make_hints(r: dict) -> list[str]:
    h = _hp()["hints"]
    out = []
    if r["type"] == "quiet":
        out.append(f"Quiet interval: Dst stayed above -30 nT (minimum {r['min_dst']:.0f} nT).")
    else:
        out.append(f"{r['cls'].capitalize()} geomagnetic storm: Dst minimum {r['min_dst']:.0f} nT.")
        main = (r["min_time"] - r["core_start"]) / pd.Timedelta(hours=1)
        out.append(f"Dst first fell below -50 nT {hrs(main)} before reaching its minimum.")
        rec = r["recovery_hours"]
        if r["recovered"]:
            out.append(f"Dst was back above -30 nT {hrs(rec)} after the minimum.")
        else:
            out.append(
                f"Dst had not recovered above -30 nT by the end of the {hrs(rec)} after the minimum covered by this window."
            )

    g = g_level(r["kp_max"])
    out.append(
        f"Kp reached {r['kp_max']:.1f}"
        + (f" (approximately NOAA {g} level)." if g else " (below G1).")
    )

    sb = h["sustained_southward_bz"]
    n = int(r["hours_bz_le_-5"])
    if n >= sb["min_hours"]:
        out.append(
            f"Sustained southward Bz: {hrs(n)} at or below -5 nT; minimum hourly Bz {r['bz_min']:.1f} nT."
        )
    elif n > 0:
        out.append(
            f"Brief southward Bz: {hrs(n)} at or below -5 nT; minimum hourly Bz {r['bz_min']:.1f} nT."
        )
    else:
        out.append(f"No southward Bz at or below -5 nT; minimum hourly Bz {r['bz_min']:.1f} nT.")
    if r["bz_min"] <= h["strongly_southward_bz_nT"]:
        out.append("Strongly southward Bz (hourly mean at or below -20 nT).")

    v, sp = r["v_max"], h["solar_wind_speed_km_s"]
    if v >= sp["very_fast"]:
        out.append(f"Very fast solar wind: maximum hourly speed {v:.0f} km/s.")
    elif v >= sp["fast"]:
        out.append(f"Fast solar wind: maximum hourly speed {v:.0f} km/s.")
    else:
        out.append(f"Ordinary solar wind speed: maximum hourly speed {v:.0f} km/s.")

    if r["pdyn_max"] >= h["dynamic_pressure_high_nPa"]:
        out.append(
            f"High solar wind dynamic pressure: up to {r['pdyn_max']:.1f} nPa (protons only)."
        )
    if r["vbs_max"] >= h["electric_field_strong_mV_m"]:
        out.append(f"Strong solar wind electric field: vBs up to {r['vbs_max']:.1f} mV/m.")

    f = r["f107_mean"]
    out.append(f"Solar activity {solar_level(f)} (mean F10.7 {f:.0f} sfu).")
    return out


def make_caveats(r: dict) -> list[str]:
    ct = _hp()["caveat_thresholds"]
    caveats = [
        "All values are hourly means in UTC; minute-level extremes are larger.",
        "Solar wind and magnetic field values are shifted to the time they reach Earth (a delay of roughly 25 to 100 minutes depending on solar wind speed).",
        "Bz is in GSM coordinates; negative Bz means southward.",
    ]
    p = float(r["plasma_dscovr_pct"])
    if p < ct["plasma_dscovr_pct_low"]:
        caveats.append(
            f"Only {p:.0f}% of the solar wind plasma hours come from DSCOVR; the rest are OMNI hourly values because DSCOVR plasma was flagged as unreliable or missing."
        )
    elif p < ct["plasma_dscovr_pct_partial"]:
        caveats.append(
            f"{p:.0f}% of the solar wind plasma hours come from DSCOVR; the rest are OMNI hourly values."
        )
    fl = float(r["field_dscovr_pct"])
    if fl < ct["field_dscovr_pct_below"]:
        caveats.append(
            f"Only {fl:.0f}% of the magnetic field hours come from DSCOVR; the rest are OMNI hourly values."
        )
    caveats.append(
        "Dst minimum is final."
        if r["dst_min_is_final"]
        else "Dst is provisional and may be revised by Kyoto."
    )
    if r["overlaps_prev"]:
        caveats.append("This window overlaps the previous event.")
    if r["overlaps_next"]:
        caveats.append("This window overlaps the next event.")
    if r["dst_window_min"] < r["min_dst"]:
        caveats.append(
            f"The window also contains a deeper Dst value ({int(r['dst_window_min'])} nT) that belongs to the overlapping event."
        )
    if r["marginal"]:
        caveats.append("Brief event just beyond the -50 nT threshold.")
    if r["f107_spike_in_window"]:
        caveats.append("Daily F10.7 includes a flagged single-day spike.")
    if r["kp_preliminary"]:
        caveats.append("Kp includes preliminary values.")
    return caveats


def make_card(r: dict, role: str | None = None) -> dict:
    storm_ev = r["type"] == "storm"
    f = float(r["f107_mean"])
    return {
        "event_id": r["event_id"],
        "type": r["type"],
        "class": r["cls"],
        "evaluation_role": role,
        "window": {
            "start": iso(r["start"]),
            "end": iso(r["end"]),
            "hours": int((r["end"] - r["start"]) / pd.Timedelta(hours=1)) + 1,
        },
        "dst": {
            "min_nT": int(r["min_dst"]),
            "min_time": iso(r["min_time"]),
            "first_below_minus50_time": iso(r["core_start"]) if storm_ev else None,
            "hours_from_minimum_to_back_above_minus30": int(r["recovery_hours"])
            if storm_ev
            else None,
            "minimum_is_final": bool(r["dst_min_is_final"]),
        },
        "drivers": {
            "bz_min_hourly_nT": round(float(r["bz_min"]), 1),
            "bz_min_time": iso(r["bz_min_time"]),
            "hours_bz_at_or_below_minus5": int(r["hours_bz_le_-5"]),
            "vbs_max_mV_per_m": round(float(r["vbs_max"]), 1),
            "vbs_mean_12h_before_dst_min_mV_per_m": round(float(r["vbs_mean_12h_before_min"]), 1),
            "speed_max_km_per_s": round(float(r["v_max"])),
            "density_max_per_cm3": round(float(r["n_max"]), 1),
            "dynamic_pressure_max_nPa_protons_only": round(float(r["pdyn_max"]), 1),
            "total_field_max_nT": round(float(r["bt_max"]), 1),
        },
        "geomagnetic": {
            "kp_max": round(float(r["kp_max"]), 2),
            "noaa_g_level_approx": g_level(float(r["kp_max"])),
            "ap_max": int(r["ap_max"]),
        },
        "solar": {"f107_mean_sfu": round(f), "level": solar_level(f)},
        "hints": make_hints(r),
        "data_quality": {
            "field_from_dscovr_pct": round(float(r["field_dscovr_pct"])),
            "plasma_from_dscovr_pct": round(float(r["plasma_dscovr_pct"])),
            "plasma_missing_pct": round(float(r["plasma_missing_pct"])),
            "caveats": make_caveats(r),
        },
        "claims_requiring_a_retrieved_source": CLAIMS_NEED_SOURCE,
    }


def add_card_inputs(events: pd.DataFrame, aligned: pd.DataFrame) -> pd.DataFrame:
    """Columns the cards need that are not in the event table: recovery and the window's Dst minimum."""
    ev = events.copy()
    ev["dst_end"] = [aligned["dst"].loc[e] for e in ev["end"]]
    ev["recovered"] = ev["dst_end"] > load_params()["events"]["storm"]["recovery_level_nT"]
    ev["dst_window_min"] = [aligned["dst"].loc[a:b].min() for a, b in zip(ev["start"], ev["end"])]
    return ev


def build_cards(
    events: pd.DataFrame, aligned: pd.DataFrame, roles: dict | None = None
) -> dict[str, dict]:
    ev = add_card_inputs(events, aligned)
    roles = roles or {}
    return {r["event_id"]: make_card(r, roles.get(r["event_id"])) for r in ev.to_dict("records")}


def check_cards(cards: dict[str, dict], events: pd.DataFrame, aligned: pd.DataFrame) -> list[str]:
    """Independent recalculation from the aligned table; returns ids of cards that disagree."""
    a = aligned.copy()
    a["vbs"] = a["v"] * (-a["bz"]).clip(lower=0) / 1000
    bad = []
    for r in events.to_dict("records"):
        A = a.loc[r["start"] : r["end"]]
        c = cards[r["event_id"]]
        own = (
            a["dst"].loc[r["core_start"] : r["core_end"]].min()
            if r["type"] == "storm"
            else A["dst"].min()
        )
        ok = (
            own == c["dst"]["min_nT"]
            and abs(A["kp"].max() - c["geomagnetic"]["kp_max"]) < 0.01
            and round(float(A["v"].max())) == c["drivers"]["speed_max_km_per_s"]
            and abs(A["bz"].min() - c["drivers"]["bz_min_hourly_nT"]) < 0.06
            and abs(A["vbs"].max() - c["drivers"]["vbs_max_mV_per_m"]) < 0.06
        )
        if not ok:
            bad.append(r["event_id"])
    return bad


def save_cards(cards: dict[str, dict], eval_ids: list[str], paths: Paths | None = None) -> None:
    p = paths or Paths()
    with open(p.event_cards, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(c, ensure_ascii=False) + "\n" for c in cards.values())
    (p.reference_events / "cards").mkdir(parents=True, exist_ok=True)
    for eid in eval_ids:
        with open(p.reference_events / "cards" / f"{eid}.json", "w", encoding="utf-8") as f:
            json.dump(cards[eid], f, indent=2, ensure_ascii=False)


def load_cards(path: Path | None = None, strip_role: bool = True) -> dict[str, dict]:
    """Read event_cards.jsonl. strip_role removes `evaluation_role` (decision 32)."""
    out = {}
    with open(path or Paths().event_cards, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if strip_role:
                c.pop("evaluation_role", None)
            out[c["event_id"]] = c
    return out


def flatten(card: object, prefix: str = "") -> dict[str, object]:
    """{'dst.min_nT': -52, ...} for every scalar field in a card (lists are indexed)."""
    out = {}
    if isinstance(card, dict):
        for k, v in card.items():
            out.update(flatten(v, f"{prefix}{k}."))
    elif isinstance(card, list):
        for i, v in enumerate(card):
            out.update(flatten(v, f"{prefix}{i}."))
    else:
        out[prefix[:-1]] = card
    return out


__all__ = [
    "CLAIMS_NEED_SOURCE",
    "build_cards",
    "check_cards",
    "flatten",
    "g_level",
    "load_cards",
    "make_card",
    "make_caveats",
    "make_hints",
    "save_cards",
]
