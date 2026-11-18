"""Train on FashionMNIST, evaluate, save the weights."""
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader

from common import (RESULTS, get_datasets, get_device,
                    seed_everything)
from model import NeuralNetwork


def train_one_epoch(loader, model, loss_fn, optimizer, device):
    model.train()
    total, n = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total += loss.item() * len(y)
        n += len(y)
    return total / n


def evaluate(loader, model, loss_fn, device):
    model.eval()
    total, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            total += loss_fn(pred, y).item() * len(y)
            correct += (pred.argmax(1) == y).sum().item()
            n += len(y)
    return total / n, correct / n


def main(epochs=5, lr=0.1):
    seed_everything(0)
    device = get_device()
    train, test = get_datasets()
    gen = torch.Generator().manual_seed(0)
    train_dl = DataLoader(train, batch_size=64, shuffle=True,
                          generator=gen)
    test_dl = DataLoader(test, batch_size=64)

    model = NeuralNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    print(f"device={device} lr={lr} epochs={epochs}")

    for epoch in range(1, epochs + 1):
        tr = train_one_epoch(
            train_dl, model, loss_fn, opt, device)
        te_loss, te_acc = evaluate(
            test_dl, model, loss_fn, device)
        print(f"epoch {epoch}: train loss {tr:.4f} | "
              f"test loss {te_loss:.4f} | "
              f"test acc {te_acc:.4f}")

    RESULTS.mkdir(exist_ok=True)
    torch.save(model.state_dict(), RESULTS / "model.pth")
    print(f"final test accuracy {te_acc:.4f}")
    print("saved results/model.pth")


if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    lr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
    main(epochs, lr)
