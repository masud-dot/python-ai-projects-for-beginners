"""A two-layer neural network in plain NumPy."""
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def init_weights(n_in, n_hidden, n_out, seed=42):
    """Small random weights, zero biases."""
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.standard_normal((n_in, n_hidden))
        / np.sqrt(n_in),
        "b1": np.zeros(n_hidden),
        "W2": rng.standard_normal((n_hidden, n_out))
        / np.sqrt(n_hidden),
        "b2": np.zeros(n_out),
    }


def forward(p, X):
    """Return the hidden and output activations."""
    a1 = sigmoid(X @ p["W1"] + p["b1"])
    a2 = sigmoid(a1 @ p["W2"] + p["b2"])
    return a1, a2


def loss(a2, Y):
    """Half the squared error per row, averaged over rows."""
    return 0.5 * np.mean(np.sum((a2 - Y) ** 2, axis=1))


def backward(p, X, Y, a1, a2):
    """Return the slope of the loss for every parameter."""
    n = len(X)
    d2 = (a2 - Y) * a2 * (1 - a2) / n
    d1 = (d2 @ p["W2"].T) * a1 * (1 - a1)
    return {
        "W1": X.T @ d1, "b1": d1.sum(axis=0),
        "W2": a1.T @ d2, "b2": d2.sum(axis=0),
    }


def update(p, grads, rate):
    for name in p:
        p[name] -= rate * grads[name]


def predict(p, X):
    return forward(p, X)[1].argmax(axis=1)


def accuracy(p, X, y):
    return float(np.mean(predict(p, X) == y))


def train(p, X, Y, rate, epochs, batch=100, seed=0):
    """Mini-batch gradient descent; return loss per epoch."""
    rng = np.random.default_rng(seed)
    history = []
    for _ in range(epochs):
        order = rng.permutation(len(X))
        for start in range(0, len(X), batch):
            rows = order[start:start + batch]
            a1, a2 = forward(p, X[rows])
            grads = backward(p, X[rows], Y[rows], a1, a2)
            update(p, grads, rate)
        history.append(loss(forward(p, X)[1], Y))
    return history
