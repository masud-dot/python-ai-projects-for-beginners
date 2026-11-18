import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
)

from train_model import train_and_predict

_, _, y_test, pred = train_and_predict()

acc = accuracy_score(y_test, pred)
counts = confusion_matrix(y_test, pred)
report = classification_report(y_test, pred, output_dict=True)
support = sum(report[str(d)]["support"] for d in range(10))

assert 0.90 <= acc <= 0.94, acc
assert counts.sum() == len(y_test) == 10_000
assert support == 10_000
assert (counts.sum(axis=1) == np.bincount(y_test)).all()
assert acc > 0.1135 + 0.5
assert report["macro avg"]["f1-score"] > 0.9
print("all checks passed, accuracy", round(acc, 4))
