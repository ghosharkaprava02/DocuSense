"""Extract in memory. No user document is written to disk."""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import pymupdf

MAX_BYTES = 25 * 1024 * 1024
MAX_PAGES = 500


@dataclass(frozen=True)
class Page:
    document_id: str
    filename: str
    number: int
    text: str


@dataclass(frozen=True)
class Document:
    id: str
    filename: str
    pages: tuple[Page, ...]
    warnings: tuple[str, ...]


def parse_pdf(content: bytes, filename: str) -> Document:
    if not content or len(content) > MAX_BYTES:
        raise ValueError("PDF must be nonempty and at most 25 MB.")
    digest = sha256(content).hexdigest()
    name = Path(filename.replace('\\', '/')).name
    try:
        with pymupdf.open(stream=content, filetype="pdf") as pdf:
            if pdf.needs_pass:
                raise ValueError("Password-protected PDF: upload an unlocked copy.")
            if len(pdf) > MAX_PAGES:
                raise ValueError(f"Maximum {MAX_PAGES} pages per document.")
            pages = tuple(Page(digest, name, i + 1, p.get_text(sort=True).strip())
                          for i, p in enumerate(pdf))
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Cannot extract this PDF. It may be corrupt or unsupported.") from exc
    if not any(p.text.strip() for p in pages):
        raise ValueError("No extractable text. This PDF may be scanned; OCR is required.")
    blank = sum(not p.text.strip() for p in pages)
    warnings = (f"{blank} pages have no extractable text; OCR may be required.",) if blank else ()
    return Document(digest, name, pages, warnings)
