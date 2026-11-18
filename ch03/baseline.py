from sklearn.dummy import DummyRegressor
from sklearn.metrics import root_mean_squared_error

from common import load, split

X_train, X_test, y_train, y_test = split(load())

dummy = DummyRegressor(strategy="mean")
dummy.fit(X_train, y_train)
pred = dummy.predict(X_test)

rmse = root_mean_squared_error(y_test, pred)
print(f"always predicts {pred[0] * 100_000:,.0f} dollars")
print(f"baseline RMSE {rmse:.3f}")
print(f"  = {rmse * 100_000:,.0f} dollars")
print(f"std of test prices {y_test.std():.3f}")
