"""Read PDFs page by page and keep the page numbers."""
from pathlib import Path

from pypdf import PdfReader


def clean(text):
    """Collapse every run of whitespace to one space."""
    return " ".join(text.split())


def load_pdf(path):
    """Return one record per page: source, page, text."""
    reader = PdfReader(path)
    records = []
    for number, page in enumerate(reader.pages, start=1):
        text = clean(page.extract_text() or "")
        records.append(
            {"source": Path(path).name, "page": number,
             "text": text}
        )
    return records


def load_folder(folder):
    """Load every PDF in a folder, in name order."""
    pages = []
    for path in sorted(Path(folder).glob("*.pdf")):
        pages.extend(load_pdf(path))
    return pages


def empty_pages(pages, minimum=20):
    """Pages with almost no text: likely scans, not text."""
    return [p for p in pages if len(p["text"]) < minimum]


if __name__ == "__main__":
    from common import DOCS

    found = load_folder(DOCS)
    print(len(found), "pages")
    for p in found[:2]:
        print(p["source"], p["page"], p["text"][:40])
    print("pages with no text:", len(empty_pages(found)))
