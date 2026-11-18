"""Datasets and DataLoaders: look at one batch."""
from torch.utils.data import DataLoader

from common import CLASSES, get_datasets, seed_everything

seed_everything(0)
train, test = get_datasets()
print("train images:", len(train), "test images:", len(test))

image, label = train[0]
print("one image:", tuple(image.shape), image.dtype)
print("pixel range:", image.min().item(), image.max().item())
print("label:", label, CLASSES[label])

loader = DataLoader(train, batch_size=64, shuffle=True)
X, y = next(iter(loader))
print("X batch:", tuple(X.shape), "y batch:", tuple(y.shape))
print("first 8 labels:", y[:8].tolist())
print("batches per epoch:", len(loader))
