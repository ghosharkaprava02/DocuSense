"""Lazy CPU embedding model and bounded, session-local embedding cache."""
from collections import OrderedDict
from hashlib import sha256

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("Embeddings must be a finite 2D matrix.")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


class Embedder:
    def __init__(self, model_name: str = MODEL_NAME):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device="cpu")
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def encode(self, texts: list[str], use_cache: bool = True) -> np.ndarray:
        if not texts:
            return np.empty((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)
        if not use_cache:
            return normalize(self.model.encode(texts, batch_size=32,
                             normalize_embeddings=True, show_progress_bar=False))
        keys = [sha256(f"{self.model_name}:{text}".encode()).hexdigest() for text in texts]
        missing = list(dict.fromkeys(k for k in keys if k not in self.cache))
        mapping = dict(zip(keys, texts))
        if missing:
            values = self.model.encode([mapping[k] for k in missing], batch_size=32,
                                       normalize_embeddings=True, show_progress_bar=False)
            for key, value in zip(missing, values):
                self.cache[key] = value
        result = np.vstack([self.cache[k] for k in keys])
        # Evict only after assembling results, including batches larger than the cache.
        while len(self.cache) > 10000:
            self.cache.popitem(last=False)
        return normalize(result)
