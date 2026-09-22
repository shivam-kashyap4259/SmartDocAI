from io import BytesIO
from pathlib import Path

from pypdf import PdfReader


def extract_pdf(file_bytes, filename):
    reader = PdfReader(BytesIO(file_bytes))

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text() or ""

        text = text.strip()

        if text:
            pages.append({
                "text": text,
                "source": filename,
                "page": page_number
            })

    return pages


def extract_txt(file_bytes, filename):

    text = file_bytes.decode(
        "utf-8",
        errors="ignore"
    ).strip()

    if not text:
        return []

    return [{
        "text": text,
        "source": filename,
        "page": None
    }]


def extract_document(file_bytes, filename):

    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        return extract_pdf(file_bytes, filename)

    elif extension == ".txt":
        return extract_txt(file_bytes, filename)

    else:
        raise ValueError(
            "Only PDF and TXT files are supported."
        )