from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

from common import SEED, load, preprocessor, split


def candidates():
    return {
        "baseline": DummyRegressor(strategy="mean"),
        "linear": LinearRegression(),
        "forest": RandomForestRegressor(
            n_estimators=100, random_state=SEED, n_jobs=-1
        ),
    }


def evaluate():
    X_train, X_test, y_train, y_test = split(load())
    prep = preprocessor(X_train.columns)
    rows = {}
    for name, est in candidates().items():
        pipe = make_pipeline(clone(prep), est)
        cv = -cross_val_score(
            pipe, X_train, y_train, cv=5,
            scoring="neg_root_mean_squared_error",
        )
        pipe.fit(X_train, y_train)
        test = root_mean_squared_error(
            y_test, pipe.predict(X_test)
        )
        rows[name] = (cv.mean(), cv.std(), test)
    return rows


if __name__ == "__main__":
    print("model      cv mean  cv std    test")
    for name, (mean, std, test) in evaluate().items():
        print(f"{name:9s} {mean:8.3f} {std:7.3f} {test:7.3f}")
