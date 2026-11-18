"""Print the sigmoid and its slope at a few inputs."""
import numpy as np

from network import sigmoid

for x in [-10, -5, -2, 0, 2, 5, 10]:
    s = sigmoid(x)
    print(f"x={x:>4}  sigmoid={s:.5f}  slope={s * (1 - s):.5f}")

print("sigmoid(0, 0) as a vector:", sigmoid(np.zeros(2)))
