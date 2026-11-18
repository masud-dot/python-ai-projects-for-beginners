"""Pick the rate on a validation split; report test once."""
from common import load, split, one_hot
from network import init_weights, train, accuracy

X, y = load()
X_train, X_test, y_train, y_test = split(X, y)
X_fit, y_fit = X_train[:55_000], y_train[:55_000]
X_val, y_val = X_train[55_000:], y_train[55_000:]

for rate in [1.0, 2.0, 5.0]:
    p = init_weights(784, 64, 10)
    train(p, X_fit, one_hot(y_fit), rate, epochs=10)
    print(f"rate {rate}: validation "
          f"{accuracy(p, X_val, y_val):.4f}")
