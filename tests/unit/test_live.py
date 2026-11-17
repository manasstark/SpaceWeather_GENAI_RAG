import copy
import json

import pandas as pd
import pytest
import requests

from spacewx_rag.live import build_live_card, get_live_data, hourly_table
from spacewx_rag.live.fetch import _active
from spacewx_rag.retrieval import event_queries
from spacewx_rag.utils.config import Paths

SAMPLE = Paths().root / "data" / "live" / "sample_snapshot.json"


@pytest.fixture(scope="module")
def snap():
    return json.loads(SAMPLE.read_text())


def test_only_the_active_monitor_is_kept():
    rows = [
        {"time_tag": "t", "active": True, "source": "SOLAR1", "bt": 1, "extra": 9},
        {"time_tag": "t", "active": False, "source": "ACE", "bt": 2},
    ]
    assert _active(rows, ["time_tag", "source", "bt"]) == [
        {"time_tag": "t", "source": "SOLAR1", "bt": 1}
    ]


def test_sample_snapshot_gives_a_valid_card(snap):
    card, table = build_live_card(snap)
    assert card["event_id"].startswith("LIVE-")
    assert card["window"]["hours"] == 24
    assert card["type"] in ("quiet", "active", "storm")
    assert "evaluation_role" not in card
    cav = " ".join(card["data_quality"]["caveats"])
    assert "not validated" in cav and "not DSCOVR" in cav and "quick-look" in cav
    assert card["data_quality"]["solar_wind_monitor"] == "SOLAR1"
    assert {"dst", "kp", "bz", "v", "n", "bt"} <= set(table.columns)


def test_solar_wind_is_shifted_to_earth_arrival(snap):
    table = hourly_table(snap)
    last_l1 = pd.to_datetime(snap["mag"][-1]["time_tag"]).tz_localize(None)
    assert table["bz"].dropna().index.max() >= last_l1.floor(
        "h"
    )  # arrives later than measured at L1


def _with_dst(snap, value):
    s = copy.deepcopy(snap)
    for row in s["dst"][-10:]:
        row["dst"] = value
    return s


def test_storm_and_active_classification(snap):
    storm, _ = build_live_card(_with_dst(snap, -120))
    assert storm["type"] == "storm" and storm["class"] == "intense"
    assert storm["dst"]["min_nT"] == -120
    assert any("not recovered" in c for c in storm["data_quality"]["caveats"])
    active, _ = build_live_card(_with_dst(snap, -40))
    assert active["type"] == "active"
    assert active["hints"][0].startswith("Active interval below storm level")
    q = event_queries(active)
    assert "physics" in q and not any(k.startswith("impacts") for k in q)  # Kp below G1


def test_falls_back_to_snapshot_when_noaa_is_down(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise requests.ConnectionError("no network")

    monkeypatch.setattr("spacewx_rag.live.fetch.fetch_feeds", boom)
    p = Paths()
    p.root = tmp_path
    (tmp_path / "data" / "live").mkdir(parents=True)
    (tmp_path / "data" / "live" / "sample_snapshot.json").write_text(SAMPLE.read_text())
    snap, status = get_live_data(p)
    assert status.startswith("offline") and snap["wind"]


def test_scenario_cards():
    from spacewx_rag.evaluation.claims import unmatched_dates
    from spacewx_rag.retrieval import overlaps_event_documents
    from spacewx_rag.scenario import PRESETS, build_scenario_card, kp_label

    may = build_scenario_card(**PRESETS["May 2024-like super storm"])
    assert may["type"] == "storm" and may["class"] == "super"
    assert may["geomagnetic"]["noaa_g_level_approx"] == "G5" and may["geomagnetic"]["ap_max"] == 400
    assert may["window"]["start"] is None and may["dst"]["min_time"] is None
    assert not any(h.startswith(("Dst first fell", "Dst was back")) for h in may["hints"])
    assert not overlaps_event_documents(may)  # no May 2024 documents for a scenario
    assert unmatched_dates("on May 10, 2024", may) == ["May 10, 2024"]
    assert build_scenario_card(**PRESETS["Quiet day"])["type"] == "quiet"
    assert build_scenario_card(**PRESETS["Unsettled"])["type"] == "active"
    odd = build_scenario_card(kp=9, dst=-10, speed=500, bz=-3)
    assert any("unusual" in c for c in odd["data_quality"]["caveats"])
    assert kp_label(5.33) == "5+" and kp_label(5.67) == "6-" and kp_label(5) == "5o"
