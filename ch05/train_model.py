"""Train the chapter's network once and save it."""
import time

import numpy as np

from common import RESULTS, load, split, one_hot
from network import init_weights, train, accuracy

SETTINGS = dict(hidden=64, rate=5.0, epochs=10)


def get_model():
    """Return the trained weights; train on the first call."""
    path = RESULTS / "model.npz"
    if path.exists():
        return dict(np.load(path))
    X, y = load()
    X_train, _, y_train, _ = split(X, y)
    p = init_weights(784, SETTINGS["hidden"], 10)
    train(p, X_train, one_hot(y_train), SETTINGS["rate"],
          SETTINGS["epochs"])
    RESULTS.mkdir(exist_ok=True)
    np.savez(path, **p)
    return p


if __name__ == "__main__":
    start = time.time()
    p = get_model()
    X, y = load()
    X_train, X_test, y_train, y_test = split(X, y)
    print(f"train accuracy {accuracy(p, X_train, y_train):.4f}")
    print(f"test accuracy {accuracy(p, X_test, y_test):.4f}")
    print(f"took {time.time() - start:.1f} seconds")
