"""Softmax plus cross-entropy on the same 784-64-10 network."""
import numpy as np

from common import load, split, one_hot
from network import init_weights, forward, update, accuracy
from network import sigmoid

X, y = load()
X_train, X_test, y_train, y_test = split(X, y)
Y = one_hot(y_train)


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def train_ce(p, rate, epochs=5, batch=100):
    rng = np.random.default_rng(0)
    for _ in range(epochs):
        order = rng.permutation(len(X_train))
        for s in range(0, len(order), batch):
            r = order[s:s + batch]
            a1 = sigmoid(X_train[r] @ p["W1"] + p["b1"])
            probs = softmax(a1 @ p["W2"] + p["b2"])
            d2 = (probs - Y[r]) / len(r)
            d1 = (d2 @ p["W2"].T) * a1 * (1 - a1)
            update(p, {"W1": X_train[r].T @ d1,
                       "b1": d1.sum(axis=0),
                       "W2": a1.T @ d2,
                       "b2": d2.sum(axis=0)}, rate)


def test_acc(p):
    a1 = sigmoid(X_test @ p["W1"] + p["b1"])
    scores = a1 @ p["W2"] + p["b2"]
    return float(np.mean(scores.argmax(axis=1) == y_test))


for hidden, rate in [(64, 0.5), (64, 5.0), (64, 10.0),
                     (128, 5.0)]:
    p = init_weights(784, hidden, 10)
    train_ce(p, rate)
    print(f"hidden {hidden:>3}, rate {rate:>4}: "
          f"test {test_acc(p):.4f}")
