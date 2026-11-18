# Chapter 5 tests. They use a small slice of MNIST, so they
# run fast.
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import network  # noqa: E402
from grad_check import worst_gap  # noqa: E402


@pytest.fixture(scope="module")
def digits():
    import common
    try:
        X, y = common.load()
    except Exception as exc:
        pytest.skip(f"MNIST unavailable: {exc}")
    return X[:500], y[:500]


def test_sigmoid_values():
    assert network.sigmoid(0) == 0.5
    assert network.sigmoid(50) > 0.999999
    assert network.sigmoid(-50) < 0.000001


def test_shapes():
    p = network.init_weights(784, 16, 10)
    a1, a2 = network.forward(p, np.zeros((3, 784)))
    assert a1.shape == (3, 16) and a2.shape == (3, 10)


def test_same_seed_same_weights():
    a = network.init_weights(784, 16, 10, seed=1)
    b = network.init_weights(784, 16, 10, seed=1)
    assert np.array_equal(a["W1"], b["W1"])


def test_gradients_match_numbers(digits):
    from common import one_hot
    X, y = digits
    X, Y = X[:10], one_hot(y[:10])
    p = network.init_weights(784, 6, 10)
    a1, a2 = network.forward(p, X)
    grads = network.backward(p, X, Y, a1, a2)
    assert max(worst_gap(p, X, Y, grads).values()) < 1e-6


def test_training_lowers_loss(digits):
    from common import one_hot
    X, y = digits
    p = network.init_weights(784, 16, 10)
    history = network.train(p, X, one_hot(y), 2.0, epochs=5)
    assert history[-1] < history[0]
    assert network.accuracy(p, X, y) > 0.3


def test_one_hot():
    from common import one_hot
    assert one_hot(np.array([3]))[0].tolist() == [
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
