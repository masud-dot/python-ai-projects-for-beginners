"""Shared loading and splitting for chapter 4."""
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml

N_TRAIN = 60_000
RESULTS = Path("results")
CACHE = Path(__file__).parent / "data" / "mnist_uint8.npz"


def load():
    """Return pixels scaled to 0-1 and integer labels.

    The first call parses the OpenML file (about 19 seconds) and
    saves the pixels as bytes; later calls read that small file.
    """
    if not CACHE.exists():
        mnist = fetch_openml(
            "mnist_784", version=1, as_frame=False,
            parser="liac-arff",
        )
        CACHE.parent.mkdir(exist_ok=True)
        np.savez(CACHE, X=mnist.data.astype(np.uint8),
                 y=mnist.target.astype(np.uint8))
    saved = np.load(CACHE)
    X = saved["X"].astype(np.float32) / 255
    return X, saved["y"].astype(int)


def split(X, y):
    """First 60,000 rows train; last 10,000 test."""
    return X[:N_TRAIN], X[N_TRAIN:], y[:N_TRAIN], y[N_TRAIN:]
