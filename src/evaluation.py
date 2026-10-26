"""Page or chunk relevance, macro Recall@K and first-relevant reciprocal rank."""
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from src.retrieval import Hit


@dataclass(frozen=True)
class Example:
    question: str
    relevant_ids: frozenset[str]
    unit: str = "page"


def evidence_id(hit: Hit, unit: str) -> str:
    if unit == "chunk":
        return hit.chunk.id
    if unit == "page":
        return f"{hit.chunk.document_id}:p{hit.chunk.page}"
    raise ValueError("Relevance unit must be page or chunk.")


def ranking_metrics(ranked: list[str], relevant: set[str] | frozenset[str]) -> dict[str, float]:
    if not relevant:
        raise ValueError("Each question needs at least one verified relevant ID.")
    # Several chunks from one page occupy actual ranks but count as one relevant page.
    result = {f"recall@{k}": len(set(ranked[:k]) & relevant) / len(relevant) for k in (1, 3, 5)}
    result["mrr@5"] = next((1 / i for i, r in enumerate(ranked[:5], 1) if r in relevant), 0.0)
    return result


def evaluate(examples: list[Example], search: Callable,
             valid_ids: dict[str, set[str]]) -> tuple[pd.DataFrame, dict]:
    if not examples:
        raise ValueError("Add verified labels before evaluating.")
    rows = []
    for example in examples:
        if not example.question.strip() or example.unit not in valid_ids:
            raise ValueError("Each row requires a question and a valid relevance unit.")
        if not example.relevant_ids or not example.relevant_ids <= valid_ids[example.unit]:
            raise ValueError("Labels reference missing or stale IDs. Re-annotate against this corpus.")
        hits, latency = search(example.question, 5)
        ranked = [evidence_id(h, example.unit) for h in hits]
        rows.append({"question": example.question, "unit": example.unit,
                     **ranking_metrics(ranked, example.relevant_ids), "retrieval_ms": latency,
                     "retrieved_ids": "|".join(ranked),
                     "relevant_ids": "|".join(sorted(example.relevant_ids))})
    frame = pd.DataFrame(rows)
    summary = frame[["recall@1", "recall@3", "recall@5", "mrr@5", "retrieval_ms"]].mean().to_dict()
    return frame, summary
