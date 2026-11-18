# Needs the FashionMNIST download, so test_ch06.py skips it.
"""Which classes does the model mix up? Counts, not guesses."""
import torch
from torch.utils.data import DataLoader

from common import (CLASSES, RESULTS, get_datasets, get_device,
                    seed_everything)
from model import NeuralNetwork
from train import main as train_main

if not (RESULTS / "model.pth").exists():
    train_main()    # a fresh machine has no saved model yet

seed_everything(0)
device = get_device()
_, test = get_datasets()
model = NeuralNetwork().to(device)
path = RESULTS / "model.pth"
model.load_state_dict(torch.load(path, weights_only=True))
model.eval()

counts = torch.zeros(10, 10, dtype=torch.long)
with torch.no_grad():
    for X, y in DataLoader(test, batch_size=256):
        guess = model(X.to(device)).argmax(1).cpu()
        for true, pred in zip(y.tolist(), guess.tolist()):
            counts[true, pred] += 1

shirt = CLASSES.index("Shirt")
print("The 1,000 Shirt images were labeled:")
row = counts[shirt]
for c in row.argsort(descending=True).tolist():
    if row[c] > 0:
        print(f"  {CLASSES[c]:12s} {row[c].item():4d}")

off = counts.clone()
off.fill_diagonal_(0)
print("Most common mix-ups (true -> predicted):")
for flat in off.flatten().argsort(descending=True)[:5].tolist():
    t, p = divmod(flat, 10)
    print(f"  {CLASSES[t]} -> {CLASSES[p]}: {off[t, p].item()}")
