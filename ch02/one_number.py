def f(x):
    return (x - 3) ** 2  # lowest point is at x = 3


def slope(x):
    return 2 * (x - 3)


x = 0.0
rate = 0.25
for step in range(6):
    print(f"step {step}  x = {x:.4f}  f(x) = {f(x):.4f}")
    x = x - rate * slope(x)
