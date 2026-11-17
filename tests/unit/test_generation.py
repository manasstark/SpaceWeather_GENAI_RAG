import pytest

from spacewx_rag.cards import load_cards
from spacewx_rag.generation import BaselineGenerator, EventDescription, user_prompt
from spacewx_rag.generation.generators import card_paths, constrained_schema
from spacewx_rag.generation.prompts import NO_PASSAGES, format_passages
from spacewx_rag.retrieval import event_queries, overlaps_event_documents


@pytest.fixture(scope="module")
def cards():
    return load_cards(strip_role=False)


def test_prompt_never_shows_evaluation_role(cards):
    c = cards["E216"]
    assert c["evaluation_role"] is not None
    p = user_prompt(c, [])
    assert "evaluation_role" not in p and "headline" not in p
    assert NO_PASSAGES in p


def test_passage_formatting_has_ids():
    s = format_passages([{"chunk_id": "d03_c000", "title": "T", "section": "S", "text": " body "}])
    assert s == "[d03_c000] (T - S)\nbody"


def test_constrained_schema_limits_citations(cards):
    c = cards["E002"]
    s = constrained_schema(c, [{"chunk_id": "d03_c000"}, {"chunk_id": "d07_c006"}])
    assert s["$defs"]["Impact"]["properties"]["sources"]["items"]["enum"] == [
        "d03_c000",
        "d07_c006",
    ]
    assert "dst.min_nT" in s["$defs"]["Characteristic"]["properties"]["evidence"]["items"]["enum"]
    assert not any(p.startswith(("evaluation_role", "claims_requiring")) for p in card_paths(c))
    assert (
        constrained_schema(c, [])["$defs"]["Interpretation"]["properties"]["sources"]["maxItems"]
        == 0
    )


def test_baseline_is_valid_and_extractive(cards):
    c = cards["E216"]
    passages = [
        {
            "chunk_id": "d03_c000",
            "title": "t",
            "section": "",
            "queries": ["physics"],
            "text": "A geomagnetic storm is a major disturbance of Earth's magnetosphere caused by solar wind energy.",
        },
        {
            "chunk_id": "d07_c006",
            "title": "t",
            "section": "",
            "queries": ["impacts: power systems"],
            "text": "Geomagnetically induced currents can flow in power grid transformers during storms.",
        },
    ]
    r = BaselineGenerator().generate(c, passages)
    out = EventDescription.model_validate(r.output)
    assert out.possible_physical_interpretation[0].statement in passages[0]["text"]
    assert out.potential_impacts[0].area == "power systems"
    assert out.summary.startswith("Super geomagnetic storm: Dst minimum -406 nT.")


def test_quiet_baseline_has_no_impacts(cards):
    r = BaselineGenerator().generate(cards["E078"], [])
    assert r.output["potential_impacts"] == []


def test_event_queries(cards):
    q = event_queries(cards["E216"])
    assert sum(k.startswith("impacts: ") for k in q) == 5 and "quality" in q
    assert "quality" not in event_queries(cards["E002"])  # final Dst
    assert not any(k.startswith("impacts") for k in event_queries(cards["E078"]))  # quiet


def test_may_2024_documents_only_for_may_2024(cards):
    assert overlaps_event_documents(cards["E216"])
    assert not overlaps_event_documents(cards["E231"])  # August 2024
    assert not overlaps_event_documents(cards["E221"])  # June 2024


def test_schema_caps_list_lengths(cards):
    s = constrained_schema(cards["E039"], [{"chunk_id": "d03_c000"}])
    assert s["$defs"]["Characteristic"]["properties"]["evidence"]["maxItems"] == 6
    assert s["$defs"]["Impact"]["properties"]["sources"]["maxItems"] == 6
