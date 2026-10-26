"""Reproducible retrieval ablation; outputs actual observations, never fixed metrics."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

from src.chunking import chunk_document
from src.embeddings import Embedder, MODEL_NAME
from src.evaluation import Example, evaluate
from src.pdf_parser import parse_pdf
from src.retrieval import SearchIndex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic", action="store_true", help="Downloads MiniLM on first run")
    args = parser.parse_args()
    corpus = ROOT / "examples" / "demo"
    docs = [parse_pdf(p.read_bytes(), p.name) for p in sorted(corpus.glob("*.pdf"))]
    raw = json.loads((corpus / "labels.json").read_text())
    examples = [Example(x["question"], frozenset(x["relevant_ids"]), x["unit"]) for x in raw]
    embedder = Embedder() if args.semantic else None
    output = ROOT / "reports"
    output.mkdir(exist_ok=True)
    summaries = []
    for size, overlap in [(80, 15), (160, 30), (220, 40)]:
        chunks = [c for d in docs for c in chunk_document(d, size, overlap)]
        valid = {"page": {f"{c.document_id}:p{c.page}" for c in chunks}, "chunk": {c.id for c in chunks}}
        for engine in ([None, embedder] if embedder else [None]):
            start = perf_counter()
            index = SearchIndex(chunks, engine)
            index_ms = (perf_counter() - start) * 1000
            frame, summary = evaluate(examples, index.search, valid)
            name = "semantic" if engine else "tfidf"
            frame.to_csv(output / f"queries_{name}_{size}.csv", index=False)
            summaries.append({"engine": name, "chunk_words": size, "overlap": overlap,
                              "chunks": len(chunks), "index_ms": index_ms,
                              "questions": len(examples), **summary})
    import pandas as pd
    pd.DataFrame(summaries).to_csv(output / "ablation.csv", index=False)
    (output / "environment.json").write_text(json.dumps({"python": sys.version,
        "platform": platform.platform(), "embedding_model": MODEL_NAME if embedder else None,
        "corpus": {d.filename: d.id for d in docs}, "seed": 42,
        "note": "Development demo set; page labels may be incomplete; no generation quality measured."}, indent=2))
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
