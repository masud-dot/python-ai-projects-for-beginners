"""Train on digits at four learning rates and plot the loss."""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import RESULTS, load, split, one_hot
from network import init_weights, train, accuracy

X, y = load()
X_train, X_test, y_train, y_test = split(X, y)
Y = one_hot(y_train)
RESULTS.mkdir(exist_ok=True)

for rate in [0.1, 1.0, 5.0, 10.0]:
    p = init_weights(784, 64, 10)
    start = time.time()
    history = train(p, X_train, Y, rate, epochs=10)
    plt.plot(range(1, 11), history, marker="o",
             label=f"rate {rate}")
    print(f"rate {rate:>4}: loss {history[0]:.4f} -> "
          f"{history[-1]:.4f}, test accuracy "
          f"{accuracy(p, X_test, y_test):.4f}, "
          f"{time.time() - start:.0f} s")

plt.xlabel("epoch")
plt.ylabel("training loss")
plt.legend()
plt.savefig(RESULTS / "learning_curves.png", dpi=150)
