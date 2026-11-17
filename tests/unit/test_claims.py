import copy

import pytest

from spacewx_rag.cards import load_cards
from spacewx_rag.evaluation.claims import (
    card_numbers,
    check_output,
    numbers,
    unmatched_dates,
    unmatched_numbers,
)
from spacewx_rag.evaluation.metrics import reference_checks


@pytest.fixture(scope="module")
def card():
    return load_cards()["E002"]


@pytest.fixture(scope="module")
def quiet_card():
    return load_cards()["E078"]


PASSAGE = {
    "chunk_id": "d03_c000",
    "title": "Geomagnetic Storms",
    "section": "",
    "text": "A geomagnetic storm is a major disturbance of Earth's magnetosphere. Storms reach 55 degrees.",
}


def test_numbers_skip_labels_ids_and_dates():
    assert numbers(
        "Kp 5.0 (G1), F10.7 77 sfu, source d03_c001 on 2016-08-03T15:00Z, Dst -52 nT"
    ) == [5.0, 77.0, -52.0]
    assert numbers("on May 10, 2024 and 3-4 August 2016") == []


def test_numbers_match_card_sign_insensitive(card):
    allowed = card_numbers(card)
    assert unmatched_numbers("Dst fell to -52 nT; 52 nT below zero", allowed) == []
    assert unmatched_numbers("Dst fell to -75 nT", allowed) == [-75.0]


def test_written_dates_checked_against_window(card):
    assert unmatched_dates("on August 3, 2016", card) == []
    assert unmatched_dates("on August 9, 2016", card) == ["August 9, 2016"]
    assert unmatched_dates("from August 2 to 3, 2016", card) == []
    assert unmatched_dates("from August 2 to August 3", card) == []
    assert unmatched_dates("2-3 August 2016 and August 3", card) == []
    assert unmatched_dates("on August 5", card) == ["August 5"]
    assert numbers("from March 13 to March 14, 2022, lasting 27 hours") == [27.0]


def good_output(card):
    return {
        "event_id": card["event_id"],
        "summary": "A moderate geomagnetic storm with a Dst minimum of -52 nT. Kp reached 5.0.",
        "characteristics": [{"statement": "Dst minimum -52 nT", "evidence": ["dst.min_nT"]}],
        "possible_physical_interpretation": [
            {"statement": "A geomagnetic storm is a major disturbance.", "sources": ["d03_c000"]}
        ],
        "potential_impacts": [],
        "caveats": [],
        "unsupported_or_unknown": [],
    }


def test_good_output_is_grounded(card):
    rows = check_output(good_output(card), card, [PASSAGE], judge=lambda c, p: (True, "ok"))
    assert all(r["grounded"] for r in rows)


def test_each_failure_mode_is_caught(card):
    o = copy.deepcopy(good_output(card))
    o["summary"] = "A coronal mass ejection caused this storm."  # cause from nowhere
    o["characteristics"][0] = {
        "statement": "Dst minimum -75 nT",
        "evidence": ["dst.min_nT"],
    }  # wrong number
    o["characteristics"].append(
        {"statement": "Kp 5.0", "evidence": ["dst.not_a_field"]}
    )  # bad evidence
    o["possible_physical_interpretation"][0]["sources"] = ["d99_c999"]  # fabricated citation
    o["potential_impacts"] = [
        {"area": "aurora", "statement": "Aurora could reach 55 degrees.", "sources": ["d03_c000"]}
    ]
    rows = {
        r["text"]: r for r in check_output(o, card, [PASSAGE], judge=lambda c, p: (False, "no"))
    }
    assert not rows["A coronal mass ejection caused this storm."]["no_cause"]
    assert rows["Dst minimum -75 nT"]["bad_numbers"] == [-75.0]
    assert not rows["Kp 5.0"]["evidence_ok"]
    assert rows["A geomagnetic storm is a major disturbance."]["fabricated_sources"] == ["d99_c999"]
    imp = rows["Aurora could reach 55 degrees."]
    assert imp["numbers_ok"]  # 55 is in the cited passage
    assert not imp["grounded"]  # but the judge said not supported
    assert not any(r["grounded"] for r in rows.values())


def test_impact_on_quiet_interval_is_ungrounded(quiet_card):
    o = good_output(quiet_card)
    o["summary"] = "Quiet conditions."
    o["characteristics"] = []
    o["potential_impacts"] = [
        {
            "area": "aurora",
            "statement": "A geomagnetic storm is a major disturbance.",
            "sources": ["d03_c000"],
        }
    ]
    rows = check_output(o, quiet_card, [PASSAGE], judge=lambda c, p: (True, "ok"))
    assert not next(r for r in rows if r["kind"] == "impact")["grounded"]


def test_reference_checks():
    ref = {
        "must_include": [
            {"id": "dst", "type": "regex", "pattern": r"-52", "section": "all"},
            {"id": "kp", "type": "regex", "pattern": r"Kp\D{0,5}6", "section": "all"},
        ],
        "must_not": [
            {"id": "no_impacts", "type": "empty", "section": "potential_impacts"},
            {"id": "cause", "type": "regex", "pattern": "coronal", "section": "summary"},
        ],
    }
    out = {
        "summary": "Dst -52 nT.",
        "characteristics": [],
        "possible_physical_interpretation": [],
        "potential_impacts": [{"statement": "x"}],
        "caveats": [],
        "unsupported_or_unknown": [],
    }
    r = reference_checks(out, ref)
    assert (
        r["completeness"] == 0.5
        and r["missed"] == ["kp"]
        and r["must_not_violations"] == ["no_impacts"]
    )


def test_impact_language_in_card_only_sections_is_ungrounded(card):
    o = good_output(card)
    o["summary"] = "A moderate storm with Dst -52 nT, potentially affecting power grids."
    rows = check_output(o, card, [PASSAGE], judge=lambda c, p: (True, "ok"))
    s = next(r for r in rows if r["kind"] == "summary")
    assert not s["no_impact"] and not s["grounded"]


def test_self_contradiction_flag():
    from spacewx_rag.evaluation.claims import self_contradiction

    o = {
        "potential_impacts": [{"statement": "x"}],
        "unsupported_or_unknown": ["impacts on satellites: no supporting source"],
    }
    assert self_contradiction(o)
    assert not self_contradiction({**o, "unsupported_or_unknown": ["cause: no supporting source"]})


def test_judge_quote_must_be_in_passages():
    from spacewx_rag.evaluation.judge import quote_found

    p = [
        {
            "text": "Power systems: widespread voltage control problems can occur,\nsome grid systems may collapse."
        }
    ]
    assert quote_found("widespread voltage control problems can occur, some grid systems", p)
    assert not quote_found("aurora visible up to 55 degrees latitude during storms", p)
    assert not quote_found("can occur", p)  # too short to count


def test_g5_effect_cannot_support_a_g4_storm(card):
    from spacewx_rag.evaluation.claims import quote_g_level

    scales = {
        "chunk_id": "d01_c000",
        "title": "NOAA scales",
        "section": "",
        "text": "G 5 Extreme Power systems: some grid systems may experience complete collapse or blackouts. "
        "G 4 Severe Power systems: possible widespread voltage control problems.",
    }
    assert (
        quote_g_level("some grid systems may experience complete collapse or blackouts", [scales])
        == 5
    )
    assert quote_g_level("possible widespread voltage control problems", [scales]) == 4
    o = good_output(card)  # E002 is G1
    o["potential_impacts"] = [
        {
            "area": "power systems",
            "statement": "Grid collapse could occur.",
            "sources": ["d01_c000"],
        }
    ]
    judge = lambda c, p: (
        True,
        "ok",
        "some grid systems may experience complete collapse or blackouts",
    )
    imp = next(r for r in check_output(o, card, [scales], judge) if r["kind"] == "impact")
    assert imp["source_g_level"] == 5 and not imp["level_ok"] and not imp["grounded"]
