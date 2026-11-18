"""Split page text into overlapping chunks of words."""
from common import CHUNK_WORDS, OVERLAP_WORDS


def chunk_words(words, size, overlap):
    """Cut a word list into windows that overlap."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("need 0 <= overlap < size")
    step = size - overlap
    pieces = []
    for start in range(0, len(words), step):
        pieces.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return pieces


def chunk_pages(pages, size=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    """Chunk each page on its own so page numbers stay exact."""
    chunks = []
    for page in pages:
        words = page["text"].split()
        for text in chunk_words(words, size, overlap):
            chunks.append(
                {"id": len(chunks), "source": page["source"],
                 "page": page["page"], "text": text}
            )
    return chunks


if __name__ == "__main__":
    demo = [{"source": "demo.pdf", "page": 1,
             "text": " ".join(f"w{i}" for i in range(1, 31))}]
    for c in chunk_pages(demo, size=12, overlap=4):
        print(c["id"], c["text"])
