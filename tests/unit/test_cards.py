import pandas as pd
import pytest

from spacewx_rag.cards import flatten, g_level, make_hints
from spacewx_rag.cards.builder import make_caveats

T = pd.Timestamp


def row(**kw):
    r = {
        "type": "storm",
        "cls": "moderate",
        "min_dst": -60,
        "min_time": T("2020-01-01 10:00"),
        "core_start": T("2020-01-01 06:00"),
        "recovery_hours": 8,
        "recovered": True,
        "kp_max": 5.33,
        "hours_bz_le_-5": 7,
        "bz_min": -9.0,
        "v_max": 450.0,
        "pdyn_max": 3.0,
        "vbs_max": 4.0,
        "f107_mean": 120.0,
        "plasma_dscovr_pct": 100.0,
        "field_dscovr_pct": 100.0,
        "dst_min_is_final": True,
        "overlaps_prev": False,
        "overlaps_next": False,
        "dst_window_min": -60,
        "marginal": False,
        "f107_spike_in_window": False,
        "kp_preliminary": False,
    }
    r.update(kw)
    return r


@pytest.mark.parametrize(
    "kp,expected",
    [(4.67, None), (4.99, "G1"), (5.0, "G1"), (6.67, "G2"), (7.0, "G3"), (8.33, "G4"), (9.0, "G5")],
)
def test_g_level_thresholds(kp, expected):
    assert g_level(kp) == expected


def test_storm_hints_text_and_rules():
    h = make_hints(row())
    assert h[0] == "Moderate geomagnetic storm: Dst minimum -60 nT."
    assert "Dst first fell below -50 nT 4 hours before reaching its minimum." in h
    assert "Kp reached 5.3 (approximately NOAA G1 level)." in h
    assert any(x.startswith("Sustained southward Bz: 7 hours") for x in h)
    assert any(x.startswith("Ordinary solar wind speed") for x in h)
    assert h[-1] == "Solar activity moderate (mean F10.7 120 sfu)."


@pytest.mark.parametrize(
    "hours,word", [(6, "Sustained"), (5, "Brief"), (1, "Brief"), (0, "No southward")]
)
def test_southward_bz_hint_boundaries(hours, word):
    assert any(x.startswith(word) for x in make_hints(row(**{"hours_bz_le_-5": hours})))


@pytest.mark.parametrize(
    "v,word", [(599, "Ordinary"), (600, "Fast"), (799, "Fast"), (800, "Very fast")]
)
def test_speed_hint_boundaries(v, word):
    assert any(x.startswith(word + " solar wind") for x in make_hints(row(v_max=float(v))))


def test_quiet_hint_and_no_storm_language():
    h = make_hints(row(type="quiet", cls="quiet", min_dst=-12, kp_max=2.0, **{"hours_bz_le_-5": 0}))
    assert h[0].startswith("Quiet interval")
    assert not any("storm" in x.lower() for x in h)


def test_caveats_follow_data_quality():
    c = make_caveats(row(plasma_dscovr_pct=29.0, dst_min_is_final=False, overlaps_next=True))
    assert any(x.startswith("Only 29% of the solar wind plasma") for x in c)
    assert "Dst is provisional and may be revised by Kyoto." in c
    assert "This window overlaps the next event." in c
    assert not any("plasma" in x for x in make_caveats(row()))


def test_flatten_paths():
    f = flatten({"dst": {"min_nT": -52}, "hints": ["a", "b"]})
    assert f == {"dst.min_nT": -52, "hints.0": "a", "hints.1": "b"}
