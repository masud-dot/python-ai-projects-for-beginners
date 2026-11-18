from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    learning_curve, validation_curve,
)
from sklearn.pipeline import make_pipeline

from common import SEED, load, preprocessor, split

SCORING = "neg_root_mean_squared_error"
Path("results").mkdir(exist_ok=True)


def report(xs, tr, va, label, name):
    """Print the curves as a table and save them as a plot."""
    train, valid = -tr.mean(axis=1), -va.mean(axis=1)
    print(f"{label:>6s}  train  valid")
    for x, a, b in zip(xs, train, valid):
        print(f"{x:6d}  {a:.3f}  {b:.3f}")
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(xs, train, marker="o", label="train")
    ax.plot(xs, valid, marker="o", label="validation")
    ax.set_xlabel(label)
    ax.set_ylabel("RMSE ($100,000s)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"results/{name}.png", dpi=150)


X_train, _, y_train, _ = split(load())
forest = RandomForestRegressor(
    n_estimators=30, random_state=SEED, n_jobs=-1
)
pipe = make_pipeline(preprocessor(X_train.columns), forest)

sizes, tr, va = learning_curve(
    pipe, X_train, y_train, cv=3, scoring=SCORING,
    train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0],
    shuffle=True, random_state=SEED,
)
report(sizes, tr, va, "rows", "learning_curve")

depths = [1, 2, 4, 6, 8, 12, 16, 24]
tr, va = validation_curve(
    pipe, X_train, y_train, cv=3, scoring=SCORING,
    param_name="randomforestregressor__max_depth",
    param_range=depths,
)
report(depths, tr, va, "depth", "validation_curve")
