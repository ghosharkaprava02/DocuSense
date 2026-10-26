"""Dense and explicitly labeled TF-IDF baselines share one result contract."""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.chunking import Chunk
from src.embeddings import Embedder, normalize


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


def cosine_rank(query: np.ndarray, matrix: np.ndarray, k: int) -> list[tuple[int, float]]:
    if k < 1:
        raise ValueError("k must be positive.")
    if len(matrix) == 0:
        return []
    q = normalize(np.asarray(query).reshape(1, -1))
    m = normalize(matrix)
    if q.shape[1] != m.shape[1]:
        raise ValueError("Query and index dimensions differ.")
    scores = (m @ q.T).ravel()
    order = np.argsort(-scores, kind="stable")[:k]
    return [(int(i), float(scores[i])) for i in order]


class SearchIndex:
    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None):
        if not chunks:
            raise ValueError("No chunks to index.")
        self.chunks = chunks
        self.embedder = embedder
        self.mode = "semantic" if embedder is not None else "keyword (TF-IDF)"
        self.vectorizer = None
        if embedder is not None:
            self.vectors = embedder.encode([c.text for c in chunks])
        else:
            self.vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True,
                                            token_pattern=r"(?u)\b\w+\b")
            try:
                self.sparse = self.vectorizer.fit_transform(c.text for c in chunks)
            except ValueError as exc:
                raise ValueError("No searchable terms after English stopword removal. "
                                 "Upload documents containing meaningful text.") from exc
            self.vectors = None

    def search(self, question: str, k: int = 5) -> tuple[list[Hit], float]:
        start = perf_counter()
        if k < 1:
            raise ValueError("k must be positive.")
        if not question.strip():
            return [], (perf_counter() - start) * 1000
        if self.embedder is not None:
            ranked = cosine_rank(self.embedder.encode([question], use_cache=False)[0], self.vectors, k)
        else:
            scores = (self.sparse @ self.vectorizer.transform([question]).T).toarray().ravel()
            order = np.argsort(-scores, kind="stable")[:k]
            ranked = [(int(i), float(scores[i])) for i in order if scores[i] > 0]
        return [Hit(self.chunks[i], score) for i, score in ranked], (perf_counter() - start) * 1000
