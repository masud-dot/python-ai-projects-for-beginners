"""Extend the project: ReLU in the hidden layer."""
import numpy as np

import network
from common import load, split, one_hot
from network import init_weights, sigmoid


def forward(p, X):
    a1 = np.maximum(X @ p["W1"] + p["b1"], 0)
    return a1, sigmoid(a1 @ p["W2"] + p["b2"])


def backward(p, X, Y, a1, a2):
    n = len(X)
    d2 = (a2 - Y) * a2 * (1 - a2) / n
    d1 = (d2 @ p["W2"].T) * (a1 > 0)
    return {
        "W1": X.T @ d1, "b1": d1.sum(axis=0),
        "W2": a1.T @ d2, "b2": d2.sum(axis=0),
    }


if __name__ == "__main__":
    X, y = load()
    X_train, X_test, y_train, y_test = split(X, y)
    Y = one_hot(y_train)
    for name, fwd, bwd in [
        ("sigmoid", network.forward, network.backward),
        ("relu", forward, backward),
    ]:
        network.forward, network.backward = fwd, bwd
        p = init_weights(784, 64, 10)
        history = network.train(p, X_train, Y, 1.0, 5)
        a2 = fwd(p, X_test)[1]
        acc = float(np.mean(a2.argmax(axis=1) == y_test))
        loss = history[-1]
        print(f"{name:<8} loss {loss:.4f}, test {acc:.4f}")
