import pytest
import pymupdf

from src.pdf_parser import parse_pdf


def test_extract_metadata(document, pdf_bytes):
    assert len(document.pages) == 2
    assert document.pages[1].number == 2
    assert "Evaluation" in document.pages[1].text
    assert document.id == parse_pdf(pdf_bytes, "renamed.pdf").id


@pytest.mark.parametrize("content", [b"", b"this is not a PDF", b"%PDF-corrupt"])
def test_invalid_pdf(content):
    with pytest.raises(ValueError):
        parse_pdf(content, "bad.pdf")


def test_scanned_or_empty():
    doc = pymupdf.open()
    doc.new_page()
    with pytest.raises(ValueError, match="OCR"):
        parse_pdf(doc.tobytes(), "blank.pdf")


def test_blank_page_warning(pdf_bytes):
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    doc.new_page()
    result = parse_pdf(doc.tobytes(), "mixed.pdf")
    assert result.warnings and "1 pages" in result.warnings[0]


def test_password_protected(pdf_bytes):
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    encrypted = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256,
                            owner_pw="owner", user_pw="reader")
    with pytest.raises(ValueError, match="Password"):
        parse_pdf(encrypted, "locked.pdf")


def test_filename_is_basename(pdf_bytes):
    assert parse_pdf(pdf_bytes, "../../private/report.pdf").filename == "report.pdf"
