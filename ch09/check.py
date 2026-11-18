"""Compare the model with my hand labels."""
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from common import RESULTS

LABELS = ["NEGATIVE", "POSITIVE"]


def check(df):
    """Return accuracy and a 2x2 confusion matrix."""
    acc = accuracy_score(df["human_label"], df["label"])
    cm = confusion_matrix(
        df["human_label"], df["label"], labels=LABELS
    )
    return acc, cm


if __name__ == "__main__":
    df = pd.read_csv(RESULTS / "scored.csv")
    acc, cm = check(df)
    print(f"accuracy {acc:.3f} on {len(df)} reviews")
    table = pd.DataFrame(
        cm,
        index=[f"human {x}" for x in LABELS],
        columns=[f"model {x}" for x in LABELS],
    )
    print(table.to_string())
    wrong = df[df["human_label"] != df["label"]]
    print("\nReviews the model got wrong:")
    for _, row in wrong.iterrows():
        short = row["text"][:50]
        print(f"{row['id']:>3} {row['score']:.2f} {short}")
