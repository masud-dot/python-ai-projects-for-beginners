import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

from common import RESULTS, load, split

raw = fetch_openml(
    "mnist_784", version=1, as_frame=False,
    parser="liac-arff",
)
print("X shape:", raw.data.shape)
print("raw label dtype:", raw.target.dtype, raw.target[:5])
print("first label == 5:", raw.target[0] == 5)
print("first label == '5':", raw.target[0] == "5")

X, y = load()
X_train, X_test, y_train, y_test = split(X, y)
print("train:", X_train.shape, "test:", X_test.shape)
print("raw pixel range:", raw.data.min(), "to", raw.data.max())
print("scaled pixel range:", X.min(), "to", X.max())
print("train n:", np.bincount(y_train))
print("test n: ", np.bincount(y_test))

RESULTS.mkdir(exist_ok=True)
fig, axes = plt.subplots(2, 10, figsize=(10, 2.4))
for digit, col in zip(range(10), axes.T):
    rows = np.flatnonzero(y_train == digit)[:2]
    for ax, i in zip(col, rows):
        ax.imshow(X_train[i].reshape(28, 28), cmap="gray_r")
        ax.set_title(str(y_train[i]), fontsize=9)
        ax.axis("off")
fig.savefig(RESULTS / "digits_grid.png", dpi=150)
print("saved results/digits_grid.png")
