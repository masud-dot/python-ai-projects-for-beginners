"""Shared helpers for Chapter 7."""
import random
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

HERE = Path(__file__).parent
DATA = HERE / "data"
RESULTS = HERE / "results"
URL = ("https://download.pytorch.org/tutorial/"
       "hymenoptera_data.zip")
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.accelerator.is_available():
        return torch.accelerator.current_accelerator().type
    return "cpu"


def get_data():
    """Download and unzip ants and bees once."""
    root = DATA / "hymenoptera_data"
    if not root.exists():
        DATA.mkdir(exist_ok=True)
        zpath = DATA / "hymenoptera_data.zip"
        print("downloading", URL)
        urllib.request.urlretrieve(URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(DATA)
    return root


def make_transform(train):
    if train:
        steps = [
            v2.RandomResizedCrop(224),
            v2.RandomHorizontalFlip(),
        ]
    else:
        steps = [v2.Resize(256), v2.CenterCrop(224)]
    steps += [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(MEAN, STD),
    ]
    return v2.Compose(steps)


def get_loaders(root, batch_size=16):
    sets = {
        name: datasets.ImageFolder(
            Path(root) / name, make_transform(name == "train"))
        for name in ("train", "val")
    }
    loaders = {
        name: DataLoader(ds, batch_size=batch_size,
                         shuffle=(name == "train"))
        for name, ds in sets.items()
    }
    return sets, loaders
