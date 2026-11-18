import numpy as np

from common import SEED, TARGET, load, preprocessor, split

df = load()
X_train, X_test, y_train, y_test = split(df)
print("train:", X_train.shape, " test:", X_test.shape)
print("same seed, same split:", end=" ")
again = split(df)[0]
print(X_train.index.equals(again.index))

prep = preprocessor(X_train.columns)
prep.fit(X_train)
scaler = prep.named_transformers_["num"][-1]
print("scaler learned from train only:",
      np.allclose(scaler.mean_, X_train.mean()))
print("features:", X_train.shape[1])
print("target:", TARGET)
