"""Compare the NumPy network with Chapter 4's SGD classifier."""
import time

from sklearn.linear_model import SGDClassifier

from common import load, split
from network import accuracy
from train_model import get_model

X, y = load()
X_train, X_test, y_train, y_test = split(X, y)

start = time.time()
sgd = SGDClassifier(random_state=42).fit(X_train, y_train)
print(f"sgd accuracy         {sgd.score(X_test, y_test):.4f}")
print(f"sgd fit took         {time.time() - start:.1f} seconds")

p = get_model()
print(f"numpy net accuracy   {accuracy(p, X_test, y_test):.4f}")
