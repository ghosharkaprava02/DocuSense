"""Descriptive statistics and exploratory, reproducible embedding clustering."""
from collections import Counter
import re

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import silhouette_score

from src.chunking import Chunk
from src.pdf_parser import Document


def keywords(text: str, n: int = 20) -> list[tuple[str, int]]:
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return Counter(w for w in words if w not in ENGLISH_STOP_WORDS).most_common(n)


def document_stats(documents: list[Document], chunks: list[Chunk]) -> pd.DataFrame:
    rows = [{"Document": d.filename, "Pages": len(d.pages),
             "Words": sum(len(p.text.split()) for p in d.pages),
             "Chunks": sum(c.document_id == d.id for c in chunks)} for d in documents]
    return pd.DataFrame(rows, columns=["Document", "Pages", "Words", "Chunks"])


def cluster_chunks(chunks: list[Chunk], vectors: np.ndarray, k: int) -> dict:
    if len(chunks) < 3 or vectors.ndim != 2 or len(vectors) != len(chunks):
        raise ValueError("At least 3 aligned chunk embeddings are required.")
    distinct = len(np.unique(vectors, axis=0))
    if not 2 <= k <= min(len(chunks) - 1, distinct):
        raise ValueError("Choose 2 through min(distinct embeddings, chunks - 1) clusters.")
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(vectors)
    coordinates = PCA(n_components=2, random_state=42).fit_transform(vectors)
    unique = len(np.unique(labels))
    score = None
    if 2 <= unique < len(vectors):
        # Stratified sample retains every cluster for a valid bounded estimate.
        rng = np.random.default_rng(42)
        ids = np.concatenate([rng.choice(np.flatnonzero(labels == i),
                              size=min(200, int((labels == i).sum())), replace=False)
                              for i in np.unique(labels)])
        if len(ids) > unique:
            score = float(silhouette_score(vectors[ids], labels[ids], metric="cosine"))
    topics = []
    for i in range(k):
        members = np.flatnonzero(labels == i)
        nearest = members[np.argmin(np.linalg.norm(vectors[members] - model.cluster_centers_[i], axis=1))]
        topics.append({"cluster": i, "size": len(members),
                       "keywords": keywords(" ".join(chunks[j].text for j in members), 8),
                       "representative": chunks[int(nearest)]})
    return {"labels": labels, "coordinates": coordinates, "silhouette": score, "topics": topics}
