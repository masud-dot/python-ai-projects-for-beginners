XS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
YS = [1.2, 2.9, 5.1, 7.2, 8.8, 11.1, 13.0, 15.2, 16.9, 19.1]


def predict(w, b, x):
    return w * x + b


def loss(w, b, xs=XS, ys=YS):
    """Mean squared error of the line y = w * x + b."""
    errors = [predict(w, b, x) - y for x, y in zip(xs, ys)]
    return sum(e * e for e in errors) / len(errors)


def gradient(w, b, xs=XS, ys=YS):
    """Slopes of the loss with respect to w and b."""
    n = len(xs)
    errors = [predict(w, b, x) - y for x, y in zip(xs, ys)]
    dw = 2 * sum(e * x for e, x in zip(errors, xs)) / n
    db = 2 * sum(errors) / n
    return dw, db


def fit(rate=0.01, steps=1000, report=200):
    w, b = 0.0, 0.0
    for step in range(steps + 1):
        if report and step % report == 0:
            print(f"step {step:4d}  loss {loss(w, b):10.4f}")
        dw, db = gradient(w, b)
        w -= rate * dw
        b -= rate * db
    return w, b


if __name__ == "__main__":
    w, b = fit()
    print(f"w = {w:.4f}, b = {b:.4f}")
