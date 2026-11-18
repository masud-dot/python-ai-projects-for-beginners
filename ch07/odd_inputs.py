"""Feed the model images with no ant or bee in them."""
import sys

import numpy as np
from PIL import Image

from common import HERE, RESULTS
from predict import load, predict

OUT = HERE / "odd"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(0)
noise = rng.integers(0, 256, (300, 300, 3), dtype=np.uint8)
Image.fromarray(noise).save(OUT / "noise.png")
gray = Image.new("RGB", (300, 300), (128, 128, 128))
gray.save(OUT / "gray.png")
Image.new("L", (300, 300), 128).save(OUT / "gray_L.png")

name = sys.argv[1] if len(sys.argv) > 1 else "finetune"
model, classes = load(RESULTS / f"{name}.pth")
for photo in sorted(OUT.glob("*.png")):
    guesses = predict(photo, model, classes, top=1)
    c, p = guesses[0]
    print(f"{photo.name}: {c} {p:.1%}")
