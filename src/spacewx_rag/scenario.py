"""What-if scenarios: an event card built from user-chosen values (the app's simulator tab).

The card uses the same classification, G-level and hint rules as real events. A scenario has no
real dates, so time fields and timing hints are removed, and the caveats say the values are
hypothetical. Derived values (vBs, dynamic pressure, |B|, ap) are computed from the inputs.
"""

from __future__ import annotations

import math

import pandas as pd

from spacewx_rag.cards.builder import make_card
from spacewx_rag.events.detection import classify
from spacewx_rag.utils.config import load_params

# official 3-hourly Kp -> ap conversion (Bartels)
KP_TO_AP = {
    0: 0,
    0.33: 2,
    0.67: 3,
    1: 4,
    1.33: 5,
    1.67: 6,
    2: 7,
    2.33: 9,
    2.67: 12,
    3: 15,
    3.33: 18,
    3.67: 22,
    4: 27,
    4.33: 32,
    4.67: 39,
    5: 48,
    5.33: 56,
    5.67: 67,
    6: 80,
    6.33: 94,
    6.67: 111,
    7: 132,
    7.33: 154,
    7.67: 179,
    8: 207,
    8.33: 236,
    8.67: 300,
    9: 400,
}
KP_STEPS = sorted(KP_TO_AP)
TIMING_HINTS = ("Dst first fell", "Dst was back", "Dst had not recovered")

PRESETS = {
    "Quiet day": {
        "kp": 1.33,
        "dst": -8,
        "speed": 360,
        "bz": -2.0,
        "hours_south": 0,
        "density": 4.0,
        "f107": 90,
    },
    "Unsettled": {
        "kp": 4.0,
        "dst": -35,
        "speed": 520,
        "bz": -6.0,
        "hours_south": 3,
        "density": 6.0,
        "f107": 140,
    },
    "Moderate storm": {
        "kp": 6.0,
        "dst": -75,
        "speed": 600,
        "bz": -12.0,
        "hours_south": 8,
        "density": 10.0,
        "f107": 130,
    },
    "Intense storm": {
        "kp": 8.0,
        "dst": -180,
        "speed": 750,
        "bz": -22.0,
        "hours_south": 18,
        "density": 15.0,
        "f107": 180,
    },
    "May 2024-like super storm": {
        "kp": 9.0,
        "dst": -406,
        "speed": 1000,
        "bz": -39.0,
        "hours_south": 23,
        "density": 30.0,
        "f107": 220,
    },
    "Carrington-like": {
        "kp": 9.0,
        "dst": -900,
        "speed": 1200,
        "bz": -50.0,
        "hours_south": 30,
        "density": 40.0,
        "f107": 250,
    },
}


def kp_label(kp: float) -> str:
    """Kp in the usual thirds notation, e.g. 5.33 -> '5+', 5.67 -> '6-'."""
    base = math.floor(kp + 1e-6)
    frac = round(kp - base, 2)
    if frac >= 0.66:
        return f"{base + 1}-"
    return f"{base}+" if frac >= 0.33 else f"{base}o"


def unusual(kp: float, dst: float, speed: float, bz: float) -> list[str]:
    """Combinations that rarely occur together, so the model is told rather than misled."""
    notes = []
    if kp >= 7 and dst > -50:
        notes.append(
            f"Kp {kp:g} with Dst above -50 nT is unusual: strong storms normally drive Dst well below -50 nT."
        )
    if kp <= 3 and dst <= -100:
        notes.append(
            f"Dst of {dst:g} nT with Kp only {kp:g} is unusual: such a depressed Dst usually comes with high Kp."
        )
    if dst <= -100 and bz > -5:
        notes.append(
            "An intense Dst without southward Bz is unusual: sustained southward Bz is the main storm driver."
        )
    if speed < 400 and dst <= -250:
        notes.append(
            "A super storm with slow solar wind is unusual: the largest storms come with fast wind."
        )
    return notes


def build_scenario_card(
    kp: float,
    dst: float,
    speed: float,
    bz: float,
    hours_south: int = 6,
    density: float = 5.0,
    f107: float = 120,
) -> dict:
    ev = load_params()["events"]
    if dst <= ev["storm"]["dst_threshold_nT"]:
        typ, cls = "storm", str(classify(pd.Series([dst])).iloc[0])
    elif dst > ev["quiet"]["dst_above_nT"] and kp <= ev["quiet"]["kp_max"]:
        typ, cls = "quiet", "quiet"
    else:
        typ, cls = "active", "active"
    vbs = speed * max(-bz, 0) / 1000
    t = pd.Timestamp("2000-01-02 00:00")  # placeholder: removed from the card below
    r = {
        "event_id": "SCENARIO",
        "type": typ,
        "cls": cls,
        "start": t - pd.Timedelta(hours=24),
        "end": t + pd.Timedelta(hours=23),
        "min_dst": dst,
        "min_time": t,
        "core_start": t - pd.Timedelta(hours=6),
        "recovery_hours": 23,
        "recovered": True,
        "bz_min": bz,
        "bz_min_time": t,
        "hours_bz_le_-5": int(hours_south if bz <= -5 else 0),
        "vbs_max": vbs,
        "vbs_mean_12h_before_min": vbs * min(hours_south / 12, 1.0),
        "v_max": speed,
        "n_max": density,
        "pdyn_max": 1.67e-6 * density * speed**2,
        "bt_max": round(math.hypot(bz, 5.0), 1),
        "kp_max": kp,
        "ap_max": KP_TO_AP[min(KP_STEPS, key=lambda k: abs(k - kp))],
        "f107_mean": f107,
        "f107_spike_in_window": False,
        "field_dscovr_pct": 100,
        "plasma_dscovr_pct": 100,
        "plasma_missing_pct": 0,
        "dst_min_is_final": False,
        "overlaps_prev": False,
        "overlaps_next": False,
        "dst_window_min": dst,
        "marginal": False,
        "kp_preliminary": False,
    }
    card = make_card(r)
    card["event_id"] = "SCENARIO"
    card.pop("evaluation_role", None)
    card["window"] = {
        "start": None,
        "end": None,
        "hours": None,
        "note": "hypothetical scenario, no real dates",
    }
    for k in ("min_time", "first_below_minus50_time", "hours_from_minimum_to_back_above_minus30"):
        card["dst"][k] = None
    card["drivers"]["bz_min_time"] = None
    card["hints"] = [h for h in card["hints"] if not h.startswith(TIMING_HINTS)]
    card["data_quality"] = {
        "source": "user-chosen scenario",
        "caveats": [
            "Hypothetical scenario: the values were set by a user, not observed.",
            "There are no real dates, so nothing can be said about a specific historical event.",
            "Bz is in GSM coordinates; negative Bz means southward.",
            "The electric field (vBs), dynamic pressure, total field and ap are derived from the chosen values.",
            *unusual(kp, dst, speed, bz),
        ],
    }
    return card
