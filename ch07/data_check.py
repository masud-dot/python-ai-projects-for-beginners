"""Look at the ants-and-bees folders and one batch."""
from common import get_data, get_loaders

root = get_data()
sets, loaders = get_loaders(root)
for name, ds in sets.items():
    counts = {c: 0 for c in ds.classes}
    for _, label in ds.samples:
        counts[ds.classes[label]] += 1
    print(name, len(ds), counts)
print("class_to_idx:", sets["train"].class_to_idx)
x, y = next(iter(loaders["train"]))
print("batch:", tuple(x.shape), x.dtype, y.tolist())
print("pixel range: %.2f to %.2f" % (x.min(), x.max()))
