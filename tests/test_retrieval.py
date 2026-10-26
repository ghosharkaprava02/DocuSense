import numpy as np
import pytest

from src.embeddings import normalize
from src.retrieval import SearchIndex, cosine_rank


def test_cosine_actual_normalization():
    ranking = cosine_rank(np.array([2, 0]), np.array([[1, 0], [0, 10], [-5, 0]]), 3)
    assert ranking == [(0, 1.0), (1, 0.0), (2, -1.0)]


def test_empty_retrieval():
    assert cosine_rank(np.array([1, 0]), np.empty((0, 2)), 5) == []


def test_zero_vectors_and_ties():
    assert np.isfinite(normalize(np.zeros((2, 3)))).all()
    assert [i for i, _ in cosine_rank(np.ones(2), np.ones((3, 2)), 2)] == [0, 1]


@pytest.mark.parametrize("matrix", [np.array([[np.nan, 0]]), np.array([1, 2])])
def test_invalid_matrix(matrix):
    with pytest.raises(ValueError):
        normalize(matrix)


def test_keyword_baseline(chunks):
    index = SearchIndex(chunks)
    hits, latency = index.search("retrieval recall", 2)
    assert hits[0].chunk.id == "c2"
    assert latency >= 0
    assert index.mode == "keyword (TF-IDF)"
    assert index.search("zzzzunknown")[0] == []
    assert index.search("  ")[0] == []


def test_dense_contract(chunks):
    class FakeEmbedder:
        def encode(self, texts, use_cache=True):
            return np.array([[1, 0], [0, 1], [-1, 0]]) if len(texts) == 3 else np.array([[1, 0]])
    index = SearchIndex(chunks, FakeEmbedder())
    assert index.search("question", 1)[0][0].chunk.id == "c1"


def test_bad_k(chunks):
    with pytest.raises(ValueError):
        SearchIndex(chunks).search("anything", 0)


def test_empty_keyword_vocabulary(chunks):
    from dataclasses import replace
    with pytest.raises(ValueError, match="No searchable terms"):
        SearchIndex([replace(chunks[0], text="the and is")])
