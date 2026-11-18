"""Appendix A, Chapter 7: build train/ and val/ from raw photos.

python split_photos.py raw_photos data/my_dataset

raw_photos holds one folder per class. Every fifth photo, in sorted
order, goes to val, so the split is the same on every run.
"""
import shutil
import sys
from pathlib import Path

KINDS = {".jpg", ".jpeg", ".png"}


def split(raw, out, every=5):
    counts = {}
    for folder in sorted(p for p in Path(raw).iterdir() if p.is_dir()):
        photos = sorted(p for p in folder.iterdir()
                        if p.suffix.lower() in KINDS)
        n_val = 0
        for i, photo in enumerate(photos):
            part = "val" if i % every == every - 1 else "train"
            n_val += part == "val"
            dest = Path(out) / part / folder.name
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(photo, dest / photo.name)
        counts[folder.name] = (len(photos) - n_val, n_val)
    return counts


if __name__ == "__main__":
    for name, (tr, va) in split(sys.argv[1], sys.argv[2]).items():
        print(f"{name}: {tr} train, {va} val")
