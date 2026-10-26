"""Page-bounded word windows; stable IDs include content and configuration."""
from dataclasses import dataclass
from hashlib import sha256

from src.pdf_parser import Document


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    filename: str
    page: int
    text: str
    start_word: int
    end_word: int


def chunk_document(document: Document, size: int = 160, overlap: int = 30) -> list[Chunk]:
    if size < 10 or not 0 <= overlap < size:
        raise ValueError("Size must be >= 10 and overlap must be between 0 and size - 1.")
    chunks = []
    for page in document.pages:
        words = page.text.split()
        for start in range(0, len(words), size - overlap):
            end = min(start + size, len(words))
            text = " ".join(words[start:end])
            key = f"{document.id}:{page.number}:{size}:{overlap}:{start}:{text}"
            chunks.append(Chunk(sha256(key.encode()).hexdigest()[:24], document.id,
                                document.filename, page.number, text, start, end))
            if end == len(words):
                break
    return chunks
