import pytest
import pymupdf

from src.chunking import Chunk
from src.pdf_parser import parse_pdf


@pytest.fixture
def pdf_bytes():
    doc = pymupdf.open()
    for text in ["Neural networks learn useful representations from data.",
                 "Evaluation measures retrieval recall and reciprocal rank."]:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    return doc.tobytes()


@pytest.fixture
def document(pdf_bytes):
    return parse_pdf(pdf_bytes, "sample.pdf")


@pytest.fixture
def chunks():
    return [Chunk("c1", "doc", "sample.pdf", 1, "neural networks learn representations", 0, 4),
            Chunk("c2", "doc", "sample.pdf", 2, "retrieval evaluation measures recall rank", 0, 5),
            Chunk("c3", "doc", "sample.pdf", 3, "monitor drift after production deployment", 0, 5)]
