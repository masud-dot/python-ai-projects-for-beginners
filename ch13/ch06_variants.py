"""Appendix A, Chapter 6: one change at a time, same seed."""
import sys

sys.path.insert(0, "vendor/ch06")
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from common import (CLASSES, get_datasets, get_device,  # noqa: E402
                    seed_everything)
from model import NeuralNetwork  # noqa: E402
from train import evaluate, train_one_epoch  # noqa: E402


def deeper():
    model = NeuralNetwork()
    layers = list(model.stack)
    layers[4:4] = [nn.Linear(512, 512), nn.ReLU()]
    model.stack = nn.Sequential(*layers)
    return model


VERSIONS = {
    "baseline": (NeuralNetwork,
                 lambda p: torch.optim.SGD(p, lr=0.1)),
    "A: third hidden layer": (
        deeper, lambda p: torch.optim.SGD(p, lr=0.1)),
    "B: SGD momentum 0.9": (
        NeuralNetwork,
        lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9)),
    "B: Adam lr 1e-3": (
        NeuralNetwork, lambda p: torch.optim.Adam(p, lr=1e-3)),
}


def per_class(model, loader, device):
    right, seen = [0] * 10, [0] * 10
    model.eval()
    with torch.no_grad():
        for X, y in loader:
            pred = model(X.to(device)).argmax(1).cpu()
            for t, p in zip(y.tolist(), pred.tolist()):
                seen[t] += 1
                right[t] += t == p
    return [r / s for r, s in zip(right, seen)]


def run(name, make_model, make_opt, epochs=5):
    seed_everything(0)
    device = get_device()
    train, test = get_datasets()
    gen = torch.Generator().manual_seed(0)
    train_dl = DataLoader(train, batch_size=64, shuffle=True,
                          generator=gen)
    test_dl = DataLoader(test, batch_size=64)
    model = make_model().to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = make_opt(model.parameters())
    for _ in range(epochs):
        train_one_epoch(train_dl, model, loss_fn, opt, device)
    acc = evaluate(test_dl, model, loss_fn, device)[1]
    accs = per_class(model, test_dl, device)
    worst = min(range(10), key=accs.__getitem__)
    print(f"{name:24s} acc {acc:.4f}  weakest "
          f"{CLASSES[worst]} {accs[worst]:.3f}")


if __name__ == "__main__":
    for name, (mm, mo) in VERSIONS.items():
        run(name, mm, mo)
