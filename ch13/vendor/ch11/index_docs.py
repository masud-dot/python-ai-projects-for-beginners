"""Build the index: PDFs in docs/ to vectors in index/."""
import time

from chunker import chunk_pages
from common import CHUNK_WORDS, DOCS, INDEX, get_embed
from embed import embed_texts, save_index
from pdf_text import empty_pages, load_folder


def build(folder, embed):
    """Return (chunks, matrix, empty pages) for a PDF folder."""
    pages = load_folder(folder)
    chunks = chunk_pages(pages)
    matrix = embed_texts([c["text"] for c in chunks], embed)
    return chunks, matrix, empty_pages(pages)


if __name__ == "__main__":
    start = time.perf_counter()
    chunks, matrix, empty = build(DOCS, get_embed())
    save_index(INDEX, chunks, matrix)
    seconds = time.perf_counter() - start
    print(f"{len(chunks)} chunks of ~{CHUNK_WORDS} words")
    print(f"vectors: {matrix.shape}; {seconds:.1f} s")
    if empty:
        print(f"WARNING: {len(empty)} pages have no text")
