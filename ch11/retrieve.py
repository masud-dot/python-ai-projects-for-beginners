"""Find the chunks closest in meaning to a question."""
import numpy as np

from common import TOP_K
from embed import embed_texts, normalize


def top_k(query, matrix, k=TOP_K):
    """Return [(row, score)] best first, by cosine similarity.

    Rows of `matrix` are unit length, so a dot product is the
    cosine. The query is normalized here.
    """
    query = normalize(np.asarray(query).reshape(1, -1))[0]
    scores = matrix @ query
    best = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in best]


def search(question, chunks, matrix, embed, k=TOP_K):
    """Embed the question; return the top chunks and scores."""
    query = embed_texts([question], embed)[0]
    return [
        dict(chunks[row], score=score)
        for row, score in top_k(query, matrix, k)
    ]


if __name__ == "__main__":
    from common import INDEX, get_embed
    from embed import load_index

    chunks, matrix = load_index(INDEX)
    question = input("Question: ")
    for hit in search(question, chunks, matrix, get_embed()):
        where = f"{hit['source']} p.{hit['page']}"
        start = hit["text"][:40]
        print(f"{hit['score']:.3f}  {where}  {start}")
