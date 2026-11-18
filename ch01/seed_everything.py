"""Seed every random source the book's projects use."""
import random

import numpy as np


def seed_everything(seed=42):
    """Seed random, NumPy and (if installed) torch.

    Returns a NumPy Generator. Pass the same seed to
    scikit-learn calls as random_state=seed.
    """
    random.seed(seed)
    np.random.seed(seed)  # for older code that uses np.random
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
    return np.random.default_rng(seed)


def draw_numbers(seed):
    rng = seed_everything(seed)
    return [
        round(random.random(), 4),
        round(float(rng.random()), 4),
        round(float(np.random.rand()), 4),
    ]


if __name__ == "__main__":
    print("run 1:", draw_numbers(42))
    print("run 2:", draw_numbers(42))
    print("run 3:", draw_numbers(7))
