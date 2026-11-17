"""src/ must reproduce the notebook outputs saved in the repository exactly."""

import pandas as pd
import pytest

from spacewx_rag.utils.config import Paths

P = Paths()
needs_interim = pytest.mark.skipif(
    not (P.interim / "dscovr_mag_1min_all.parquet").exists(),
    reason="data/interim not built (run scripts/download_data.py and `spacewx interim`)",
)
needs_index = pytest.mark.skipif(
    not (P.vector_store / "chroma.sqlite3").exists(), reason="Chroma index not built"
)


@pytest.fixture(scope="module")
def aligned():
    return pd.read_parquet(P.aligned)


@pytest.fixture(scope="module")
def events():
    return pd.read_parquet(P.event_table)


@needs_interim
def test_alignment_reproduces_notebook_02(aligned):
    from spacewx_rag.data import build_aligned

    pd.testing.assert_frame_equal(build_aligned(), aligned, check_freq=False)


def test_event_table_reproduces_notebook_03(aligned, events):
    from spacewx_rag.events import build_event_table

    pd.testing.assert_frame_equal(build_event_table(aligned), events)


def test_cards_reproduce_notebook_04(aligned, events):
    from spacewx_rag.cards import build_cards, check_cards, load_cards

    roles = pd.read_csv(P.evaluation_set, index_col="event_id")["role"].to_dict()
    cards = build_cards(events, aligned, roles)
    assert cards == load_cards(strip_role=False)
    assert check_cards(cards, events, aligned) == []


def test_chunks_reproduce_notebook_05(tmp_path):
    from spacewx_rag.knowledge.build import chunk_documents, load_chunks

    saved = load_chunks()
    p = Paths()
    p.chunks = tmp_path / "chunks.jsonl"
    new = chunk_documents(p)
    assert new["chunk_id"].tolist() == saved["chunk_id"].tolist()
    assert new["text"].tolist() == saved["text"].tolist()


@needs_index
def test_event_documents_never_leak():
    from spacewx_rag.cards import load_cards
    from spacewx_rag.retrieval import Retriever, overlaps_event_documents

    r, cards = Retriever(), load_cards()
    for eid in pd.read_csv(P.evaluation_set)["event_id"]:
        docs = {p["doc_id"] for p in r.passages_for_event(cards[eid])}
        assert bool(docs & {12, 13}) <= overlaps_event_documents(cards[eid]), eid
