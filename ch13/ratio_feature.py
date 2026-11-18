"""Appendix A, Chapter 3: one ratio feature."""
import sys

sys.path.insert(0, "vendor/ch03")
from sklearn.base import clone  # noqa: E402
from sklearn.metrics import root_mean_squared_error  # noqa: E402
from sklearn.model_selection import cross_val_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402

from common import load, preprocessor, split  # noqa: E402
from models import candidates  # noqa: E402

NEW = {
    "none": None,
    "BedRatio": lambda d: d["AveBedrms"] / d["AveRooms"],
    "Households": lambda d: d["Population"] / d["AveOccup"],
}


def run(name):
    df = load()
    if NEW[name] is not None:
        df[name] = NEW[name](df)
        assert df[name].notna().all() and df[name].abs().max() < 1e9
    X_train, X_test, y_train, y_test = split(df)
    prep = preprocessor(X_train.columns)
    for model in ("linear", "forest"):
        pipe = make_pipeline(clone(prep), candidates()[model])
        cv = -cross_val_score(pipe, X_train, y_train, cv=5,
                              scoring="neg_root_mean_squared_error")
        pipe.fit(X_train, y_train)
        test = root_mean_squared_error(y_test, pipe.predict(X_test))
        print(f"{name:11s} {model:7s} test {test:.3f} "
              f"cv std {cv.std():.3f}")


if __name__ == "__main__":
    for name in NEW:
        run(name)
