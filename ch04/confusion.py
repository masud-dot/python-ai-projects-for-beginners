import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay, confusion_matrix,
)

from common import RESULTS
from train_model import train_and_predict

_, _, y_test, pred = train_and_predict()
RESULTS.mkdir(exist_ok=True)

counts = confusion_matrix(y_test, pred)
print("counts sum to", counts.sum())
print("row sums:", counts.sum(axis=1))

fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test, pred, normalize="true", values_format=".2f",
    cmap="Greys", ax=ax,
)
fig.savefig(RESULTS / "confusion.png", dpi=150)

off = counts.copy()
np.fill_diagonal(off, 0)
print("top mistakes (true -> predicted: count):")
for i in np.argsort(off, axis=None)[::-1][:6]:
    t, p = divmod(i, 10)
    print(f"  {t} -> {p}: {off[t, p]}")

# Same view as a share of each true digit, not a raw count.
rates = off / counts.sum(axis=1, keepdims=True)
print("top mistake rates (share of the true digit):")
for i in np.argsort(rates, axis=None)[::-1][:3]:
    t, p = divmod(i, 10)
    print(f"  {t} -> {p}: {rates[t, p]:.3f}")
