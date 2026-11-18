"""Compare chunk sizes on one question with a known page."""
import sys

from chunker import chunk_pages
from common import OVERLAP_WORDS, get_embed
from embed import embed_texts
from pdf_text import load_folder
from retrieve import top_k

SIZES = (20, 60, 200)


def rank_of_page(question, page, pages, embed, size):
    """Chunk count and rank of the first chunk on `page`."""
    overlap = min(OVERLAP_WORDS, size // 4)
    chunks = chunk_pages(pages, size, overlap)
    matrix = embed_texts([c["text"] for c in chunks], embed)
    query = embed_texts([question], embed)[0]
    ranked = top_k(query, matrix, k=len(chunks))
    for rank, (row, _) in enumerate(ranked, start=1):
        if chunks[row]["page"] == page:
            return len(chunks), rank
    return len(chunks), None


if __name__ == "__main__":
    from common import DOCS

    question, page = sys.argv[1], int(sys.argv[2])
    pages = load_folder(DOCS)
    embed = get_embed()
    for size in SIZES:
        count, rank = rank_of_page(
            question, page, pages, embed, size
        )
        print(f"size {size:>3}: {count:>3} chunks, "
              f"right page ranks {rank}")
