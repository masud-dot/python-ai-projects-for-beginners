"""Try hidden-layer sizes and learning rates."""
import time

from common import load, split, one_hot
from network import init_weights, train, accuracy

X, y = load()
X_train, X_test, y_train, y_test = split(X, y)
Y = one_hot(y_train)

runs = [(16, 5.0), (64, 5.0), (128, 5.0), (128, 2.0)]
for hidden, rate in runs:
    p = init_weights(784, hidden, 10)
    start = time.time()
    train(p, X_train, Y, rate, epochs=10)
    n_weights = sum(w.size for w in p.values())
    acc = accuracy(p, X_test, y_test)
    print(f"hidden {hidden:>3}, rate {rate}: "
          f"{n_weights:>7,} numbers,"
          f" test {acc:.4f}, {time.time() - start:.0f} s")
