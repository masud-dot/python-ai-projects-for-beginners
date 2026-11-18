"""Shared loading for chapter 5."""
from pathlib import Path

import numpy as np

RESULTS = Path("results")
CACHE = Path(__file__).parent / "data" / "mnist_uint8.npz"


def load():
    """Return pixels scaled to 0-1 and integer labels.

    The first call downloads MNIST with scikit-learn and saves
    the pixels as bytes; later calls read that small file.
    """
    if not CACHE.exists():
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml(
            "mnist_784", version=1, as_frame=False,
            parser="liac-arff",
        )
        CACHE.parent.mkdir(exist_ok=True)
        np.savez(CACHE, X=mnist.data.astype(np.uint8),
                 y=mnist.target.astype(np.uint8))
    saved = np.load(CACHE)
    return saved["X"] / 255, saved["y"].astype(int)


def split(X, y):
    """First 60,000 rows train; last 10,000 test."""
    return X[:60_000], X[60_000:], y[:60_000], y[60_000:]


def one_hot(y, n_classes=10):
    """Turn a label like 3 into a row with one 1 at slot 3."""
    return np.eye(n_classes)[y]
