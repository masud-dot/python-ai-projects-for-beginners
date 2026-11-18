"""Smoke test: train 3 epochs on 1,000 images."""
from common import load, one_hot
from network import init_weights, train

X, y = load()
p = init_weights(784, 16, 10)
history = train(p, X[:1000], one_hot(y[:1000]), 2.0, 3)
print([round(float(h), 4) for h in history])
