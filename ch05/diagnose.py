"""Measure why rate 10.0 and 128 units at rate 5.0 fail."""
import numpy as np

from common import load, split, one_hot
from network import init_weights, forward, train, accuracy

X, y = load()
X_train, X_test, y_train, y_test = split(X, y)
Y = one_hot(y_train)

for hidden, rate in [(64, 5.0), (64, 10.0), (128, 5.0)]:
    p = init_weights(784, hidden, 10)
    train(p, X_train, Y, rate, epochs=1)
    a1, a2 = forward(p, X_train[:1000])
    dead = np.mean(a2 * (1 - a2) < 0.01)
    print(f"hidden {hidden:>3}, rate {rate:>4}: "
          f"output {a2.mean():.3f}, flat {dead:.0%}, "
          f"train {accuracy(p, X_train, y_train):.4f}")
