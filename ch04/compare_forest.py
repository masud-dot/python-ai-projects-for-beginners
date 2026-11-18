import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score

from common import load, split

X, y = load()
X_train, X_test, y_train, y_test = split(X, y)

models = {
    "sgd": SGDClassifier(random_state=42),
    "forest": RandomForestClassifier(random_state=42),
}
scores = {}
for name, model in models.items():
    start = time.perf_counter()
    model.fit(X_train, y_train)
    secs = time.perf_counter() - start
    pred = model.predict(X_test)
    scores[name] = f1_score(y_test, pred, average=None)
    print(f"{name}: accuracy {(pred == y_test).mean():.4f}, "
          f"fit {secs:.1f} s")

print("digit  sgd F1  forest F1  change")
for d in range(10):
    a, b = scores["sgd"][d], scores["forest"][d]
    print(f"{d:>5}  {a:.3f}   {b:.3f}      {b - a:+.3f}")
