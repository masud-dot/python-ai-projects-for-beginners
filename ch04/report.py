import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score,
)
from sklearn.model_selection import cross_val_predict

from common import load, split

X, y = load()
X_train, X_test, y_train, y_test = split(X, y)

model = SGDClassifier(random_state=42).fit(X_train, y_train)
pred = model.predict(X_test)
print(classification_report(y_test, pred, digits=3))

# A one-question detector: is this digit a 5?
t_train, t_test = y_train == 5, y_test == 5
print("share of 5s in test:", round(t_test.mean(), 4))

never = DummyClassifier(strategy="most_frequent")
never.fit(X_train, t_train)
p = never.predict(X_test)
print("never-5 accuracy:", round((p == t_test).mean(), 4))
print("never-5 recall:  ", recall_score(t_test, p))

det = SGDClassifier(random_state=42).fit(X_train, t_train)
p = det.predict(X_test)
print("detector accuracy: ", round((p == t_test).mean(), 4))
print("detector precision:",
      round(precision_score(t_test, p), 3))
print("detector recall:   ", round(recall_score(t_test, p), 3))
print("detector F1:       ", round(f1_score(t_test, p), 3))
print("detector counts [[TN, FP], [FN, TP]]:")
print(confusion_matrix(t_test, p))

# Move the threshold: pick it on training data, judge on test.
scores = cross_val_predict(
    SGDClassifier(random_state=42), X_train, t_train,
    cv=3, method="decision_function",
)
prec, rec, thr = precision_recall_curve(t_train, scores)
cut = thr[np.argmax(prec[:-1] >= 0.90)]
print("threshold for 90% precision (train):", round(cut, 2))

p = det.decision_function(X_test) >= cut
print("test precision:", round(precision_score(t_test, p), 3))
print("test recall:   ", round(recall_score(t_test, p), 3))
