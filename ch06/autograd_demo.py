"""Autograd: Chapter 5's neuron slope, then accumulation."""
import torch

x, target = 2.0, 1.0
w = torch.tensor(0.5, requires_grad=True)

z = w * x
a = torch.sigmoid(z)
loss = 0.5 * (a - target) ** 2
print(f"loss={loss.item():.4f} grad_fn={loss.grad_fn.name()}")

loss.backward()
print(f"w.grad = {w.grad.item():.6f}")

# Calling backward again on a fresh graph ADDS to w.grad.
for call in (2, 3):
    loss = 0.5 * (torch.sigmoid(w * x) - target) ** 2
    loss.backward()
    print(f"after backward #{call}: {w.grad.item():.6f}")

w.grad.zero_()
loss = 0.5 * (torch.sigmoid(w * x) - target) ** 2
loss.backward()
print(f"after zero_grad: {w.grad.item():.6f}")

with torch.no_grad():
    y = w * x
print("tracked inside no_grad:", y.requires_grad)
