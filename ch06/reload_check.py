"""Reload the saved weights; per-class accuracy."""
import torch
from torch import nn
from torch.utils.data import DataLoader

from common import (CLASSES, RESULTS, get_datasets, get_device,
                    seed_everything)
from model import NeuralNetwork
from train import evaluate, main as train_main

if not (RESULTS / "model.pth").exists():
    train_main()    # a fresh machine has no saved model yet

seed_everything(0)
device = get_device()
_, test = get_datasets()
test_dl = DataLoader(test, batch_size=64)

model = NeuralNetwork().to(device)
path = RESULTS / "model.pth"
model.load_state_dict(torch.load(path, weights_only=True))
loss_fn = nn.CrossEntropyLoss()
loss, acc = evaluate(test_dl, model, loss_fn, device)
print(f"reloaded test accuracy {acc:.4f} (loss {loss:.4f})")

right = torch.zeros(10)
seen = torch.zeros(10)
model.eval()
with torch.no_grad():
    for X, y in test_dl:
        guess = model(X.to(device)).argmax(1).cpu()
        for c in range(10):
            mask = y == c
            seen[c] += mask.sum()
            right[c] += (guess[mask] == c).sum()

order = (right / seen).argsort()
for c in order.tolist():
    print(f"{CLASSES[c]:12s} {right[c] / seen[c]:.3f}")
