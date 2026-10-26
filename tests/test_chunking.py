from dataclasses import replace
import pytest

from src.chunking import chunk_document
from src.pdf_parser import Page


def test_overlap_and_stable_ids(document):
    page = Page(document.id, document.filename, 1, " ".join(f"w{i}" for i in range(25)))
    doc = replace(document, pages=(page,))
    chunks = chunk_document(doc, 10, 3)
    assert len(chunks) == 4
    assert chunks[0].text.split()[-3:] == chunks[1].text.split()[:3]
    assert chunks == chunk_document(doc, 10, 3)
    assert len({c.id for c in chunks}) == len(chunks)
    assert chunks[1].start_word == 7
    assert chunks[-1].end_word == 25
    assert chunks[0].id != chunk_document(doc, 11, 3)[0].id


def test_page_metadata(document):
    result = chunk_document(document)
    assert [c.page for c in result] == [1, 2]
    assert all(c.document_id == document.id for c in result)


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, -1), (5, 1)])
def test_invalid_windows(document, size, overlap):
    with pytest.raises(ValueError):
        chunk_document(document, size, overlap)
