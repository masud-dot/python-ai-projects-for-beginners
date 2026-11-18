"""Embed chunks, save them to disk and load them back."""
import json

import numpy as np

from common import EMBED_MODEL


def normalize(matrix):
    """Scale every row to length 1 (leave zero rows alone)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def embed_texts(texts, embed, batch=16):
    """Embed texts in batches; return a unit-length matrix."""
    vectors = []
    for start in range(0, len(texts), batch):
        response = embed(
            model=EMBED_MODEL, input=texts[start:start + batch]
        )
        vectors.extend(response["embeddings"])
    return normalize(np.array(vectors, dtype=np.float32))


def save_index(folder, chunks, matrix):
    """Write vectors, chunks and the embedding model's name."""
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "vectors.npy", matrix)
    meta = {"model": EMBED_MODEL, "count": len(chunks)}
    (folder / "meta.json").write_text(json.dumps(meta))
    (folder / "chunks.json").write_text(
        json.dumps(chunks), encoding="utf-8"
    )


def load_index(folder):
    """Return (chunks, matrix); refuse a mismatched index."""
    meta = json.loads((folder / "meta.json").read_text())
    if meta["model"] != EMBED_MODEL:
        raise ValueError(
            f"index was built with {meta['model']}, but "
            f"EMBED_MODEL is {EMBED_MODEL}; rebuild it"
        )
    matrix = np.load(folder / "vectors.npy")
    text = (folder / "chunks.json").read_text(encoding="utf-8")
    return json.loads(text), matrix
