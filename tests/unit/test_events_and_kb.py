import pandas as pd
import pytest

from spacewx_rag.events import classify, episodes
from spacewx_rag.knowledge import sections


def test_episodes_merge_gap():
    m = pd.Series([0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1], dtype=bool)
    assert episodes(m, 1) == [[1, 3], [5, 6], [10, 12]]
    assert episodes(m, 3) == [[1, 6], [10, 12]]  # gap of 2 hours < 3 -> merged
    assert episodes(m, 12) == [[1, 12]]


def test_episodes_empty():
    assert episodes(pd.Series([False] * 5), 12) == []


@pytest.mark.parametrize(
    "dst,cls",
    [
        (-50, "moderate"),
        (-99, "moderate"),
        (-100, "intense"),
        (-249, "intense"),
        (-250, "super"),
        (-406, "super"),
    ],
)
def test_storm_classes(dst, cls):
    assert classify(pd.Series([dst])).astype(str).iloc[0] == cls


def test_sections_split_on_headings():
    text = "intro line\n## First\nbody one\n## Second\nbody two\nmore"
    assert sections(text) == [
        ("", "intro line"),
        ("First", "body one"),
        ("Second", "body two\nmore"),
    ]
