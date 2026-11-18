"""One sigmoid neuron: chain rule by hand versus a nudge."""
import numpy as np

from network import sigmoid

x, w, b, target = 2.0, 0.5, 0.0, 1.0


def loss_for(w):
    a = sigmoid(w * x + b)
    return 0.5 * (a - target) ** 2


z = w * x + b
a = sigmoid(z)
d_loss_d_a = a - target          # slope of the loss w.r.t. a
d_a_d_z = a * (1 - a)            # slope of the sigmoid
d_z_d_w = x                      # slope of z w.r.t. w
slope = d_loss_d_a * d_a_d_z * d_z_d_w

nudge = 1e-6
high, low = loss_for(w + nudge), loss_for(w - nudge)
numeric = (high - low) / (2 * nudge)

print(f"z={z:.4f} a={a:.4f} loss={loss_for(w):.4f}")
print(f"chain rule slope {slope:.6f}")
print(f"nudge slope      {numeric:.6f}")
new_w = w - 1.0 * slope
print(f"new w {new_w:.4f} at rate 1.0")
print(f"loss {loss_for(w):.4f} -> {loss_for(new_w):.4f}")
