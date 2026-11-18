import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

from common import SEED, load, preprocessor, split

X_train, X_test, y_train, y_test = split(load())
SCORING = "neg_root_mean_squared_error"


def build(est):
    return make_pipeline(preprocessor(X_train.columns), est)


forest = RandomForestRegressor(
    n_estimators=100, random_state=SEED, n_jobs=-1
)
search = GridSearchCV(
    build(forest),
    {"randomforestregressor__min_samples_leaf": [1, 3, 5]},
    cv=5, scoring=SCORING,
)
search.fit(X_train, y_train)
res = search.cv_results_
for p, m in zip(res["params"], -res["mean_test_score"]):
    leaf = p["randomforestregressor__min_samples_leaf"]
    print(f"min_samples_leaf {leaf}: cv RMSE {m:.3f}")
best = search.best_params_
print("best:", best)
test = root_mean_squared_error(y_test, search.predict(X_test))
print(f"test RMSE of the best forest {test:.3f}")

for name, est in [("linear", LinearRegression()),
                  ("ridge", Ridge(alpha=10.0))]:
    pipe = build(est).fit(X_train, y_train)
    rmse = root_mean_squared_error(
        y_test, pipe.predict(X_test)
    )
    print(f"{name} test RMSE {rmse:.3f}")

lin = build(LinearRegression()).fit(X_train, y_train)
weights = lin[-1].coef_
for col, w in sorted(zip(X_train.columns, weights),
                     key=lambda t: -abs(t[1])):
    print(f"{col:11s} {w:+.3f}")
