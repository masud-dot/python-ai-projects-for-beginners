"""Where the network fails: saturation and moved digits."""
import numpy as np

from common import load, split, one_hot
from network import init_weights, forward, train, accuracy
from train_model import get_model

X, y = load()
X_train, X_test, y_train, y_test = split(X, y)
Y = one_hot(y_train)


def saturated(p, X):
    """Share of hidden outputs whose slope is below 0.01."""
    a1 = forward(p, X[:1000])[0]
    return float(np.mean(a1 * (1 - a1) < 0.01))


for label, s1, s2 in [("normal start", 1, 1),
                      ("large W1 only", 20, 1),
                      ("large W1 and W2", 20, 20)]:
    p = init_weights(784, 64, 10)
    p["W1"] *= s1
    p["W2"] *= s2
    share = saturated(p, X_train)
    history = train(p, X_train[:10_000], Y[:10_000], 1.0, 5)
    acc = accuracy(p, X_test, y_test)
    print(f"{label:<16} saturated {share:.0%}, "
          f"loss {history[-1]:.3f}, test {acc:.4f}")

p = get_model()
images = X_test.reshape(-1, 28, 28)
for label, moved in [
    ("original", images),
    ("shifted 2 px right", np.roll(images, 2, axis=2)),
    ("rotated 90 degrees", np.rot90(images, axes=(1, 2))),
]:
    flat = moved.reshape(len(moved), 784)
    print(f"{label:<20}{accuracy(p, flat, y_test):.4f}")
