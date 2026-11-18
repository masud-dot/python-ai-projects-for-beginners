import time

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier

from common import load, split

X, y = load()
X_train, X_test, y_train, y_test = split(X, y)

dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
print("dummy accuracy", round(dummy.score(X_test, y_test), 4))

sgd = SGDClassifier(random_state=42)
start = time.perf_counter()
sgd.fit(X_train, y_train)
secs = time.perf_counter() - start
print("sgd accuracy  ", round(sgd.score(X_test, y_test), 4))
print(f"sgd fit took {secs:.1f} seconds")
print("first test digit: predicted", sgd.predict(X_test[:1])[0],
      "actual", y_test[0])
