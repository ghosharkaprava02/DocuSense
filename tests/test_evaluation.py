import pytest

from src.evaluation import Example, evaluate, ranking_metrics
from src.retrieval import SearchIndex


def test_known_metrics():
    metrics = ranking_metrics(["x", "a", "a", "b", "y"], {"a", "b"})
    assert metrics == {"recall@1": 0, "recall@3": .5, "recall@5": 1, "mrr@5": .5}


def test_empty_ranking():
    assert ranking_metrics([], {"a"})["mrr@5"] == 0
    with pytest.raises(ValueError):
        ranking_metrics([], set())


def test_real_index_evaluation(chunks):
    frame, summary = evaluate([Example("retrieval recall", frozenset({"doc:p2"}))],
                               SearchIndex(chunks).search, {"page": {"doc:p1", "doc:p2"}})
    assert summary["recall@1"] == 1
    assert len(frame) == 1


def test_stale_labels_rejected(chunks):
    with pytest.raises(ValueError, match="stale"):
        evaluate([Example("what?", frozenset({"old:p1"}))], SearchIndex(chunks).search,
                 {"page": {"doc:p1"}})


def test_no_labels_rejected(chunks):
    with pytest.raises(ValueError):
        evaluate([], SearchIndex(chunks).search, {})
