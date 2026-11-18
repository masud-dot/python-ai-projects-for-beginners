"""Assertions that tell you the chapter's network is right."""
import numpy as np

from common import load, split, one_hot
from grad_check import worst_gap
from network import init_weights, forward, backward, accuracy
from train_model import get_model

X, y = load()
X_train, X_test, y_train, y_test = split(X, y)

# 1. Shapes through the network.
p = init_weights(784, 64, 10)
a1, a2 = forward(p, X_train[:7])
assert a1.shape == (7, 64) and a2.shape == (7, 10)

# 2. Gradients match the numerical estimate.
small, Y_small = X_train[:20], one_hot(y_train[:20])
q = init_weights(784, 8, 10)
b1, b2 = forward(q, small)
grads = backward(q, small, Y_small, b1, b2)
gaps = worst_gap(q, small, Y_small, grads)
assert max(gaps.values()) < 1e-6, gaps

# 3. Training beats the random start by a wide margin.
trained = get_model()
before = accuracy(p, X_test, y_test)
after = accuracy(trained, X_test, y_test)
assert before < 0.3 and after > 0.95, (before, after)

print(f"all checks passed, accuracy {after:.4f}")
