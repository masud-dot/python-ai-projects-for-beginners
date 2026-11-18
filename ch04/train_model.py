"""Train the chapter's classifier and save its predictions."""
import numpy as np
from sklearn.linear_model import SGDClassifier

from common import RESULTS, load, split


def train_and_predict():
    X, y = load()
    X_train, X_test, y_train, y_test = split(X, y)
    model = SGDClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test, model.predict(X_test)


if __name__ == "__main__":
    model, X_test, y_test, pred = train_and_predict()
    RESULTS.mkdir(exist_ok=True)
    np.save(RESULTS / "pred.npy", pred)
    print("accuracy", round((pred == y_test).mean(), 4))
