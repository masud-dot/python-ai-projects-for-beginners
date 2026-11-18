"""Follow one batch through the model, layer by layer."""
import torch

from model import NeuralNetwork

model = NeuralNetwork()
x = torch.rand(64, 1, 28, 28)
print(f"{'input':10s}", tuple(x.shape))
x = model.flatten(x)
print(f"{'Flatten':10s}", tuple(x.shape))
for layer in model.stack:
    x = layer(x)
    print(f"{type(layer).__name__:10s}", tuple(x.shape))

print()
for name, p in model.named_parameters():
    print(f"{name:14s}", tuple(p.shape))
total = sum(p.numel() for p in model.parameters())
print(f"parameters: {total:,}")
