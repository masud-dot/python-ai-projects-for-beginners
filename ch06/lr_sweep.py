# Needs the FashionMNIST download, so test_ch06.py skips it.
"""Learning rate sweep: three epochs at each rate."""
import torch
from torch import nn
from torch.utils.data import DataLoader

from common import get_datasets, get_device, seed_everything
from model import NeuralNetwork
from train import evaluate, train_one_epoch

RATES = [0.001, 0.01, 0.1, 0.5, 1.0]


def run(lr, epochs=3):
    seed_everything(0)
    device = get_device()
    train, test = get_datasets()
    gen = torch.Generator().manual_seed(0)
    train_dl = DataLoader(
        train, batch_size=64, shuffle=True, generator=gen)
    test_dl = DataLoader(test, batch_size=256)
    model = NeuralNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(epochs):
        tr = train_one_epoch(
            train_dl, model, loss_fn, opt, device)
    _, acc = evaluate(test_dl, model, loss_fn, device)
    return tr, acc


if __name__ == "__main__":
    print("   lr  train loss  test acc")
    for lr in RATES:
        tr, acc = run(lr)
        print(f"{lr:5g}  {tr:10.4f}  {acc:8.4f}")
