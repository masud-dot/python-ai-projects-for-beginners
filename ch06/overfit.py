"""Overfitting: 1,000 training images, many epochs."""
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from common import get_datasets, get_device, seed_everything
from model import NeuralNetwork
from train import evaluate, train_one_epoch


def main(n_train=1000, epochs=100, every=10):
    seed_everything(0)
    device = get_device()
    train, test = get_datasets()
    small = Subset(train, range(n_train))
    gen = torch.Generator().manual_seed(0)
    train_dl = DataLoader(small, batch_size=64, shuffle=True,
                          generator=gen)
    eval_train = DataLoader(small, batch_size=256)
    test_dl = DataLoader(test, batch_size=256)

    model = NeuralNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    print("epoch  train acc  test acc  test loss")
    for epoch in range(1, epochs + 1):
        train_one_epoch(train_dl, model, loss_fn, opt, device)
        if epoch % every == 0:
            _, tr_acc = evaluate(
                eval_train, model, loss_fn, device)
            te_loss, te_acc = evaluate(
                test_dl, model, loss_fn, device)
            print(f"{epoch:5d}  {tr_acc:9.3f}  {te_acc:8.3f}"
                  f"  {te_loss:9.3f}")


if __name__ == "__main__":
    main(epochs=int(sys.argv[1]) if len(sys.argv) > 1 else 100)
