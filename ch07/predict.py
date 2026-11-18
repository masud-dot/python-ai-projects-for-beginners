"""Classify photos: python predict.py MODEL.pth photo.jpg ..."""
import sys

import torch
from PIL import Image

from common import make_transform
from model import build_model


def load(path):
    ckpt = torch.load(path, map_location="cpu",
                      weights_only=True)
    model = build_model(len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["classes"]


def predict(photo, model, classes, top=2):
    img = Image.open(photo).convert("RGB")
    x = make_transform(train=False)(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    best = probs.argsort(descending=True)[:top]
    return [(classes[i], probs[i].item()) for i in best]


if __name__ == "__main__":
    model, classes = load(sys.argv[1])
    for photo in sys.argv[2:]:
        guesses = predict(photo, model, classes)
        text = ", ".join(f"{c} {p:.1%}" for c, p in guesses)
        print(f"{photo}: {text}")
