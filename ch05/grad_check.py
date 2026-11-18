"""Check backward() against a numerical estimate."""
import numpy as np

from common import load, split, one_hot
from network import init_weights, forward, backward, loss


def numerical(p, X, Y, name, nudge=1e-5):
    """Slope of the loss for each entry of p[name]."""
    slopes = np.zeros_like(p[name])
    for i in np.ndindex(p[name].shape):
        old = p[name][i]
        p[name][i] = old + nudge
        high = loss(forward(p, X)[1], Y)
        p[name][i] = old - nudge
        low = loss(forward(p, X)[1], Y)
        p[name][i] = old
        slopes[i] = (high - low) / (2 * nudge)
    return slopes


def worst_gap(p, X, Y, grads):
    """Largest relative gap between the slopes, per array."""
    gaps = {}
    for name in p:
        mine, check = grads[name], numerical(p, X, Y, name)
        top = np.abs(mine - check).max()
        gaps[name] = top / max(np.abs(mine).max(), 1e-12)
    return gaps


if __name__ == "__main__":
    X, y = load()
    X_small, Y_small = X[:20], one_hot(y[:20])
    p = init_weights(784, 8, 10)
    a1, a2 = forward(p, X_small)
    grads = backward(p, X_small, Y_small, a1, a2)
    gaps = worst_gap(p, X_small, Y_small, grads)
    for name, gap in gaps.items():
        print(f"{name}: relative gap {gap:.2e}")

    # Break backward() on purpose: forget the sigmoid slope.
    n = len(X_small)
    wrong_d2 = (a2 - Y_small) / n
    wrong = dict(grads, W2=a1.T @ wrong_d2)
    gap = worst_gap(p, X_small, Y_small, wrong)["W2"]
    print(f"W2, factor missing: relative gap {gap:.2e}")
