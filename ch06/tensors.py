"""Tensors: create, reshape, move and convert."""
import numpy as np
import torch

a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print("a:", a.shape, a.dtype, a.device)
print("zeros:", torch.zeros(2, 2).tolist())

flat = a.reshape(6)
print("flat:", flat.tolist())
print("a * 2 + 1:", (a * 2 + 1).tolist())
print("a @ a.T:", (a @ a.T).tolist())

arr = np.array([1, 2, 3])
t = torch.from_numpy(arr)
arr[0] = 99
print("shares memory:", t.tolist())
print("int dtype:", t.dtype)
print("back to NumPy:", type(a.numpy()).__name__)

if torch.accelerator.is_available():
    kind = torch.accelerator.current_accelerator().type
else:
    kind = "cpu"
print("device:", kind)
on_device = a.to(kind)
print("moved:", on_device.device.type)
