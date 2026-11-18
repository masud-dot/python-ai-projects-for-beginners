import pandas as pd

import compare_models as cm


def test_compare_finds_disagreements():
    a = pd.DataFrame({"id": [1, 2, 3], "text": ["x", "y", "z"],
                      "human_label": ["POSITIVE", "NEGATIVE", "POSITIVE"],
                      "label": ["POSITIVE", "NEGATIVE", "NEGATIVE"]})
    b = a.copy()
    b["label"] = ["positive", "positive", "negative"]
    both, diff = cm.compare(a, b)
    assert len(both) == 3
    assert list(diff["id"]) == [2]
