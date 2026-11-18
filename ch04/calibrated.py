"""Does a probability beat the margin gap for review?"""
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict

from common import load, split

X, y = load()
X_train, X_test, y_train, y_test = split(X, y)


def gap(scores):
    top = np.sort(scores, axis=1)
    return top[:, -1] - top[:, -2]


def review(name, model, method, confidence):
    """Review the 20% least sure images; count catches."""
    cv = cross_val_predict(model, X_train, y_train, cv=5,
                           n_jobs=-1, method=method)
    cut = np.quantile(confidence(cv), 0.20)
    model.fit(X_train, y_train)
    wrong = model.predict(X_test) != y_test
    out = getattr(model, method)(X_test)
    flag = confidence(out) <= cut
    print(f"{name}: acc {1 - wrong.mean():.4f}, "
          f"reviewed {flag.sum()}, "
          f"caught {(wrong & flag).sum()} of {wrong.sum()}")


review("hinge gap", SGDClassifier(random_state=42),
       "decision_function", gap)
review("log loss",
       SGDClassifier(loss="log_loss", random_state=42),
       "predict_proba", lambda p: p.max(axis=1))
