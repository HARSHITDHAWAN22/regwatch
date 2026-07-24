"""
Extracts raw text (+ any easily-available metadata) from an uploaded
circular. Supports PDF and plain text so you can test quickly with .txt
sample circulars without needing a PDF for every test case.
"""
import pdfplumber
from pathlib import Path


def parse_document(file_path: str) -> str:
    path = Path(file_path)
    if path.suffix.lower() == ".pdf":
        return _parse_pdf(file_path)
    return path.read_text(encoding="utf-8", errors="ignore")


def _parse_pdf(file_path: str) -> str:
    text_parts = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)
