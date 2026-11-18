"""Look at the saved index: shape, lengths, page counts."""
from collections import Counter

import numpy as np

from common import INDEX
from embed import load_index


def summary(chunks, matrix):
    """Facts about an index that you can check by eye."""
    norms = np.linalg.norm(matrix, axis=1)
    pages = Counter((c["source"], c["page"]) for c in chunks)
    return {
        "shape": matrix.shape,
        "shortest": float(norms.min()),
        "longest": float(norms.max()),
        "per_page": dict(pages),
    }


def similarity(matrix, i, j):
    """Cosine of two rows (rows have length one)."""
    return float(matrix[i] @ matrix[j])


if __name__ == "__main__":
    chunks, matrix = load_index(INDEX)
    info = summary(chunks, matrix)
    print("matrix shape:", info["shape"])
    print(f"vector lengths: {info['shortest']:.3f} to "
          f"{info['longest']:.3f}")
    for (name, page), count in info["per_page"].items():
        print(f"{name} p.{page}: {count} chunks")
    print(f"chunk 0 vs chunk 1: {similarity(matrix, 0, 1):.3f}")
    print(f"chunk 0 vs last: {similarity(matrix, 0, -1):.3f}")
