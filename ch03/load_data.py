import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import TARGET, load

df = load()
print(df.shape)
print(textwrap.fill(", ".join(df.columns), 60))
print(df.isna().sum().sum(), "missing values")

stats = df.describe().T[["min", "max"]]
print(stats)
top = df[TARGET].max()
print((df[TARGET] == top).sum(), "block groups at", top)

Path("results").mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(5, 3))
ax.hist(df[TARGET], bins=50)
ax.set_xlabel("Median house value ($100,000s)")
ax.set_ylabel("Block groups")
fig.tight_layout()
fig.savefig("results/target_hist.png", dpi=150)
print("saved results/target_hist.png")
