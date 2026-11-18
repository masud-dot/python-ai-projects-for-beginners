"""Turn scored reviews into counts, means and a threshold."""
import sys

import pandas as pd

from common import RESULTS


def add_decision(df, threshold=0.9):
    """Mark reviews below the threshold as UNSURE."""
    df = df.copy()
    df["decision"] = df["label"].where(
        df["score"] >= threshold, "UNSURE"
    )
    return df


def summarize(df):
    """Return label counts, mean scores, lowest five."""
    counts = df["label"].value_counts()
    means = df.groupby("label")["score"].mean().round(3)
    lowest = df.nsmallest(5, "score")[["id", "label", "score"]]
    return counts, means, lowest


if __name__ == "__main__":
    threshold = float(sys.argv[1]) if len(sys.argv) > 1 else 0.9
    df = pd.read_csv(RESULTS / "scored.csv")
    counts, means, lowest = summarize(df)
    print("Counts by label:")
    print(counts.to_string())
    print("\nMean score by label:")
    print(means.to_string())
    print("\nFive lowest-confidence reviews:")
    print(lowest.to_string(index=False))
    decided = add_decision(df, threshold)
    print(f"\nWith threshold {threshold}:")
    print(decided["decision"].value_counts().to_string())
    decided.to_csv(RESULTS / "decided.csv", index=False)
