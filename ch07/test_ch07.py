# Chapter 7 tests. They build a tiny folder of synthetic images
# and use weights=None, so they run offline and fast.
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from common import get_loaders, make_transform  # noqa: E402
from model import build_model, count_params  # noqa: E402
from predict import load, predict  # noqa: E402
from check import evaluate, same_pipeline  # noqa: E402
from train import train  # noqa: E402


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        for cls, base in (("ants", 40), ("bees", 200)):
            d = root / split / cls
            d.mkdir(parents=True)
            for i in range(4):
                arr = rng.integers(
                    base, base + 40, (260, 300, 3), dtype=np.uint8)
                Image.fromarray(arr).save(d / f"{i}.jpg")
    return root


def test_loaders(root):
    sets, loaders = get_loaders(root, batch_size=4)
    assert sets["train"].classes == ["ants", "bees"]
    x, y = next(iter(loaders["train"]))
    assert x.shape == (4, 3, 224, 224)
    assert x.dtype == torch.float32


def test_freeze_counts():
    total, full = count_params(build_model(pretrained=False))
    _, head = count_params(
        build_model(freeze=True, pretrained=False))
    assert head == 512 * 2 + 2
    assert full == total
    assert total == 11_177_538


def test_train_predict_check(root, tmp_path):
    out = tmp_path / "m.pth"
    best, _ = train("freeze", epochs=1, root=root,
                    pretrained=False, out=out)
    assert 0.0 <= best <= 1.0
    model, classes = load(out)
    photo = next((root / "val" / "ants").glob("*.jpg"))
    guesses = predict(photo, model, classes)
    assert len(guesses) == 2
    assert sum(p for _, p in guesses) == pytest.approx(1.0, abs=1e-4)
    sets, loaders = get_loaders(root, batch_size=4)
    cm = evaluate(model, loaders["val"], 2)
    assert cm.sum().item() == 8
    assert same_pipeline(sets["val"])


def test_grayscale_photo(root, tmp_path):
    gray = tmp_path / "g.png"
    Image.new("L", (300, 300), 128).save(gray)
    model = build_model(pretrained=False).eval()
    out = predict(gray, model, ["ants", "bees"])
    assert len(out) == 2


def test_eval_transform_is_deterministic():
    img = Image.new("RGB", (300, 400), (10, 20, 30))
    tf = make_transform(train=False)
    assert torch.equal(tf(img), tf(img))
