"""One real local RAG request, saved with evidence and actual timings."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

from src.chunking import chunk_document
from src.embeddings import Embedder
from src.pdf_parser import parse_pdf
from src.rag import generate_answer
from src.retrieval import SearchIndex

if __name__ == "__main__":
    docs = [parse_pdf(p.read_bytes(), p.name) for p in sorted((ROOT / "examples/demo").glob("*.pdf"))]
    chunks = [c for doc in docs for c in chunk_document(doc)]
    index = SearchIndex(chunks, Embedder())
    question = "What is the difference between data drift and concept drift?"
    hits, latency = index.search(question, 3)
    answer = generate_answer(question, hits, timeout=180)
    report = {"question": question, "retrieval_engine": index.mode, "retrieval_ms": latency,
              "answer": asdict(answer), "evidence": [asdict(h) for h in hits],
              "limitation": "Single smoke example; source ID validation is not a factuality benchmark."}
    (ROOT / "reports" / "rag_smoke.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"retrieval_ms": latency, **asdict(answer)}, indent=2))
    sys.exit(0 if answer.status == "generated" else 1)
