"""Train: python train.py MODE [EPOCHS] [DATA_FOLDER]

MODE is finetune (all weights) or freeze (new last layer only).
DATA_FOLDER defaults to the ants-and-bees download.
"""
import sys
import time
from pathlib import Path

import torch
from torch import nn, optim

from common import (RESULTS, get_data, get_device, get_loaders,
                    seed_everything)
from model import build_model


def run_epoch(model, loader, device, loss_fn, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
    return loss_sum / total, correct / total


def train(mode, epochs=10, root=None, pretrained=True,
          out=None):
    seed_everything(0)
    device = get_device()
    root = root or get_data()
    sets, loaders = get_loaders(root)
    classes = sets["train"].classes
    model = build_model(
        len(classes), freeze=(mode == "freeze"),
        pretrained=pretrained).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=7, gamma=0.1)
    loss_fn = nn.CrossEntropyLoss()
    print(f"device={device} mode={mode} epochs={epochs}")
    best, start = 0.0, time.time()
    out = out or RESULTS / f"{mode}.pth"
    out.parent.mkdir(exist_ok=True)
    for epoch in range(1, epochs + 1):
        tl, ta = run_epoch(model, loaders["train"], device,
                           loss_fn, optimizer)
        scheduler.step()
        vl, va = run_epoch(model, loaders["val"], device,
                           loss_fn)
        print(f"epoch {epoch:2d}: train loss {tl:.4f}"
              f" acc {ta:.4f} | val loss {vl:.4f}"
              f" acc {va:.4f}")
        if va >= best:
            best = va
            state = {k: v.cpu() for k, v in
                     model.state_dict().items()}
            torch.save({"state_dict": state,
                        "classes": classes}, out)
    secs = time.time() - start
    print(f"best val acc {best:.4f} in {secs:.0f} s")
    print("saved", out)
    return best, secs


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "finetune"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    root = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    train(mode, epochs, root=root)
