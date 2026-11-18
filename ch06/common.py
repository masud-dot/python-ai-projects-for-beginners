"""Shared helpers for Chapter 6."""
import random
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets
from torchvision.transforms import v2

HERE = Path(__file__).parent
DATA = HERE / "data"
RESULTS = HERE / "results"
CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.accelerator.is_available():
        return torch.accelerator.current_accelerator().type
    return "cpu"


def get_datasets():
    prep = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    train = datasets.FashionMNIST(
        DATA, train=True, download=True, transform=prep)
    test = datasets.FashionMNIST(
        DATA, train=False, download=True, transform=prep)
    return train, test
