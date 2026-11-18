"""Run the forward pass on five digits with random weights."""
import numpy as np

from common import load, split, one_hot
from network import init_weights, forward, loss

X, y = load()
X_train, X_test, y_train, y_test = split(X, y)

p = init_weights(784, 64, 10)
batch = X_train[:5]
a1, a2 = forward(p, batch)

print("input shape   ", batch.shape)
print("W1, W2 shapes ", p["W1"].shape, p["W2"].shape)
print("hidden shape  ", a1.shape)
print("output shape  ", a2.shape)
print("first output row:")
print(np.round(a2[0], 3))
print("predicted", a2.argmax(axis=1), "actual", y_train[:5])
print("loss", round(loss(a2, one_hot(y_train[:5])), 4))
