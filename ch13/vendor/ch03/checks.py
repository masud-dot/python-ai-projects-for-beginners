from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models import evaluate
from common import SEED, load, preprocessor, split

rows = evaluate()
base, lin, forest = (rows[k][2] for k in
                     ("baseline", "linear", "forest"))
X_train, X_test, y_train, y_test = split(load())

assert abs(base - y_test.std()) < 0.05, "baseline off"
assert forest < lin < base, "models out of order"
print(f"1. order ok: {forest:.3f} < {lin:.3f} < {base:.3f}")

again = evaluate()["forest"][2]
assert again == forest, "run is not repeatable"
print("2. same seed, same test RMSE")

prep = preprocessor(X_train.columns).fit(X_train)
scaler = prep.named_transformers_["num"][-1]
assert np.allclose(scaler.mean_, X_train.mean())
assert X_train.index.intersection(X_test.index).empty
leaky = StandardScaler().fit(pd.concat([X_train, X_test]))
assert not np.allclose(leaky.mean_, X_train.mean()), "blind"
print("3. no leakage, and the check can see a leak")

pipe = make_pipeline(
    preprocessor(X_train.columns),
    RandomForestRegressor(
        n_estimators=100, random_state=SEED, n_jobs=-1
    ),
).fit(X_train, y_train)
pred = pipe.predict(X_test)
assert np.isclose(root_mean_squared_error(y_test, pred), forest)
print("4. rebuilt forest matches the table")

Path("results").mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(y_test, pred, s=4, alpha=0.3)
ax.plot([0, 5], [0, 5], color="black")
ax.set_xlabel("Actual ($100,000s)")
ax.set_ylabel("Predicted ($100,000s)")
fig.tight_layout()
fig.savefig("results/pred_vs_actual.png", dpi=150)
print("saved results/pred_vs_actual.png")
