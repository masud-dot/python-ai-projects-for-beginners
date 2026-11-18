"""Check a model: python check.py MODE [DATA_FOLDER]"""
import sys
from pathlib import Path

import torch
from PIL import Image

from common import (RESULTS, get_data, get_loaders,
                    make_transform)
from predict import load


def evaluate(model, loader, num_classes):
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    with torch.no_grad():
        for x, y in loader:
            guess = model(x).argmax(dim=1)
            for t, g in zip(y, guess):
                cm[t, g] += 1
    return cm


def same_pipeline(dataset):
    """Does the loader and predict.py see identical pixels?"""
    path, _ = dataset.samples[0]
    from_loader = dataset[0][0]
    img = Image.open(path).convert("RGB")
    from_predict = make_transform(train=False)(img)
    return torch.allclose(from_loader, from_predict)


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "finetune"
    model, classes = load(RESULTS / f"{name}.pth")
    root = get_data()
    if len(sys.argv) > 2:
        root = Path(sys.argv[2])
    sets, loaders = get_loaders(root)
    cm = evaluate(model, loaders["val"], len(classes))
    print("rows = true class, columns = predicted")
    print("        ", *[f"{c:>6}" for c in classes])
    for i, row in enumerate(cm.tolist()):
        print(f"{classes[i]:>8}", *[f"{n:6d}" for n in row])
    acc = cm.diag().sum().item() / cm.sum().item()
    print(f"val accuracy {acc:.4f}")
    print("same pixels in loader and predict:",
          same_pipeline(sets["val"]))
    print("misclassified:")
    with torch.no_grad():
        for path, label in sets["val"].samples:
            img = Image.open(path).convert("RGB")
            x = make_transform(train=False)(img).unsqueeze(0)
            if model(x).argmax().item() != label:
                print(" ", path)
