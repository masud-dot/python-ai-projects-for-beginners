"""Appendix A, Chapter 9: compare two scored files.

python compare_models.py scored_a.csv scored_b.csv

Each file is the scored.csv that score.py wrote for one model,
renamed. Labels are upper-cased, so POSITIVE and positive match.
"""
import sys

import pandas as pd


def compare(a, b):
    """Merge by id; return the table and the disagreements."""
    both = a.merge(b, on=["id", "text", "human_label"],
                   suffixes=("_a", "_b"))
    for side in ("a", "b"):
        both[f"label_{side}"] = both[f"label_{side}"].str.upper()
    both["agree"] = both["label_a"] == both["label_b"]
    return both, both[~both["agree"]]


if __name__ == "__main__":
    a, b = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
    both, diff = compare(a, b)
    n = len(both)
    for side in ("a", "b"):
        hits = (both[f"label_{side}"] == both["human_label"]).sum()
        print(f"model {side}: {hits}/{n} match the human label")
    print(f"models agree on {n - len(diff)}/{n} reviews")
    for _, r in diff.iterrows():
        print(f"{r['id']:>3} human {r['human_label']:<8} "
              f"a {r['label_a']:<8} b {r['label_b']:<8} "
              f"{r['text'][:40]}")
