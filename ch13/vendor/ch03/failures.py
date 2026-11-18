import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from common import SEED, load, preprocessor, split

X_train, X_test, y_train, y_test = split(load())


def fit(est):
    prep = preprocessor(X_train.columns)
    return make_pipeline(prep, est).fit(X_train, y_train)


forest = fit(RandomForestRegressor(
    n_estimators=100, random_state=SEED, n_jobs=-1))
linear = fit(LinearRegression())

err = forest.predict(X_test) - y_test
bands = pd.cut(y_test, [0, 1, 2, 3, 4, 5, 6], right=False)
table = err.groupby(bands, observed=True).agg(
    rows="size",
    lean="mean",
    rmse=lambda e: float(np.sqrt((e ** 2).mean())),
)
print(table.round(3))
print("prices at the cap:", int((y_test >= 5).sum()))

rich = X_test.iloc[[0]].copy()
for income in (5, 15, 30, 60):
    row = rich.copy()
    row["MedInc"] = income
    f, ln = forest.predict(row)[0], linear.predict(row)[0]
    print(f"income {income}: forest {f:.2f}  linear {ln:.2f}")
print("largest price in training data:", y_train.max().round(2))
