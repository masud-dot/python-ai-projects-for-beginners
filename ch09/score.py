"""Score a CSV of reviews and save the results."""
import sys

import pandas as pd

from common import HERE, RESULTS, load_pipe


def clean(df):
    """Drop missing or blank texts and trim the spaces."""
    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].str.strip()
    return df[df["text"] != ""]


def score(df, pipe, batch_size=1):
    """Add label and score columns, one row per review."""
    results = pipe(
        df["text"].tolist(),
        batch_size=batch_size,
        truncation=True,
    )
    df = df.copy()
    df["label"] = [r["label"] for r in results]
    df["score"] = [r["score"] for r in results]
    return df


if __name__ == "__main__":
    default = HERE / "reviews.csv"
    path = sys.argv[1] if len(sys.argv) > 1 else default
    raw = pd.read_csv(path)
    df = clean(raw)
    print(f"{len(raw)} rows read, {len(df)} kept")
    scored = score(df, load_pipe())
    RESULTS.mkdir(exist_ok=True)
    scored.to_csv(RESULTS / "scored.csv", index=False)
    print(f"saved {len(scored)} rows to results/scored.csv")
