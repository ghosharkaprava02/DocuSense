import numpy as np
import pytest

from src.analytics import cluster_chunks, document_stats, keywords


def test_stats(document, chunks):
    frame = document_stats([document], chunks)
    assert frame.iloc[0].Pages == 2
    assert frame.iloc[0].Words > 0
    assert keywords("the the embeddings embeddings model")[0] == ("embeddings", 2)


def test_clustering(chunks):
    matrix = np.array([[1., 0], [.99, .01], [0, 1]])
    result = cluster_chunks(chunks, matrix, 2)
    assert result["coordinates"].shape == (3, 2)
    assert sum(t["size"] for t in result["topics"]) == 3
    assert result["silhouette"] is not None
    assert result["labels"].tolist() == cluster_chunks(chunks, matrix, 2)["labels"].tolist()


def test_invalid_cluster_count(chunks):
    with pytest.raises(ValueError):
        cluster_chunks(chunks, np.eye(3), 3)
    with pytest.raises(ValueError):
        cluster_chunks(chunks, np.ones((3, 2)), 2)
