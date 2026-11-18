# Needs the FashionMNIST download, so test_ch06.py skips it.
"""Trace where the loss becomes nan at learning rate 1.0."""
import torch
from torch import nn
from torch.utils.data import DataLoader

from common import get_datasets, get_device, seed_everything
from model import NeuralNetwork

seed_everything(0)
device = get_device()
train, _ = get_datasets()
gen = torch.Generator().manual_seed(0)
dl = DataLoader(train, batch_size=64, shuffle=True,
                generator=gen)
model = NeuralNetwork().to(device)
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.SGD(model.parameters(), lr=1.0)

for i, (X, y) in enumerate(dl):
    X, y = X.to(device), y.to(device)
    logits = model(X)
    loss = loss_fn(logits, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if i < 5 or not torch.isfinite(loss):
        print(f"batch {i:3d} loss {loss.item():10.4f} "
              f"max |logit| {logits.abs().max().item():.1f}")
    if not torch.isfinite(loss):
        break
