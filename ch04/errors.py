import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict

from common import RESULTS, load, split
from train_model import train_and_predict

# Step 1: choose what to study from training data only.
X, y = load()
X_train, _, y_train, _ = split(X, y)
cv_scores = cross_val_predict(
    SGDClassifier(random_state=42), X_train, y_train,
    cv=5, n_jobs=-1, method="decision_function",
)
cv_pred = cv_scores.argmax(axis=1)
print("cross-validated accuracy:",
      round((cv_pred == y_train).mean(), 4))
off = np.zeros((10, 10), int)
np.add.at(off, (y_train, cv_pred), 1)
np.fill_diagonal(off, 0)
pair_total = off + off.T
ranked = np.argsort(np.triu(pair_total), axis=None)[::-1][:3]
for i in ranked:
    a, b = divmod(i, 10)
    print(f"  pair {a}/{b}: {pair_total[a, b]} mistakes")
low, high = divmod(ranked[0], 10)

model, X_test, y_test, pred = train_and_predict()
RESULTS.mkdir(exist_ok=True)

wrong = np.flatnonzero(pred != y_test)
print("mistakes:", len(wrong), "of", len(y_test))

pairs = [(low, high), (high, low)]
fig, axes = plt.subplots(2, 6, figsize=(9, 3.4))
for row, (true, guess) in zip(axes, pairs):
    hits = np.flatnonzero((y_test == true) & (pred == guess))
    print(f"true {true}, predicted {guess}: {len(hits)} images")
    for ax, i in zip(row, hits[:6]):
        ax.imshow(X_test[i].reshape(28, 28), cmap="gray_r")
        ax.set_title(f"true {true}, said {guess}", fontsize=8)
        ax.axis("off")
name = f"mistakes_{low}_{high}.png"
fig.savefig(RESULTS / name, dpi=150)
print("saved results/" + name)

# Step 2: how sure was the model? Gap between best two scores.
def gaps(scores):
    top = np.sort(scores, axis=1)
    return top[:, -1] - top[:, -2]


gap_cv = gaps(cv_scores)
gap_test = gaps(model.decision_function(X_test))
right = pred == y_test
med = np.median
print("median gap, right:", round(med(gap_test[right]), 2))
print("median gap, wrong:", round(med(gap_test[~right]), 2))

# Fix the review cut-off on training data, then use it on test.
cut = np.quantile(gap_cv, 0.20)
cv_wrong = cv_pred != y_train
print("cut-off (20% least sure in training):", round(cut, 2))
print("training: caught", (cv_wrong & (gap_cv <= cut)).sum(),
      "of", cv_wrong.sum())
flag = gap_test <= cut
print("test: reviewed", flag.sum(), "caught",
      (~right & flag).sum(), "of", len(wrong))
