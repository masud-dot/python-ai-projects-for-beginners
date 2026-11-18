# Chapter 9 tests. They run offline: the model is replaced by a
# fake pipeline. first.py, fails.py and the model download need
# the real model, so they are not run here.
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import check  # noqa: E402
import fails  # noqa: E402
import score  # noqa: E402
import summarize  # noqa: E402


def fake_pipe(texts, batch_size=1, truncation=False):
    assert truncation is True
    out = []
    for t in texts:
        bad = "broke" in t or "late" in t
        out.append({
            "label": "NEGATIVE" if bad else "POSITIVE",
            "score": 0.97 if bad else 0.62,
        })
    return out


def reviews():
    return pd.read_csv(HERE / "reviews.csv")


def test_csv_shape():
    df = reviews()
    assert len(df) == 40
    assert df["text"].isna().sum() == 1
    assert set(df["human_label"]) == {"POSITIVE", "NEGATIVE"}


def test_clean_drops_empty_and_blank():
    texts = ["a ", None, "  "]
    raw = pd.DataFrame({"id": [1, 2, 3], "text": texts})
    out = score.clean(raw)
    assert out["text"].tolist() == ["a"]


def test_score_adds_columns():
    df = score.clean(reviews())
    assert len(df) == 39
    out = score.score(df, fake_pipe)
    assert {"label", "score"} <= set(out.columns)
    assert len(out) == 39


def test_summarize_and_threshold():
    df = score.score(score.clean(reviews()), fake_pipe)
    counts, means, lowest = summarize.summarize(df)
    assert counts.sum() == 39
    assert len(lowest) == 5
    assert lowest["score"].max() <= 0.62
    dec = summarize.add_decision(df, 0.9)
    unsure = dec["decision"] == "UNSURE"
    assert (dec.loc[unsure, "score"] < 0.9).all()
    kept = dec[~unsure]
    assert (kept["label"] == kept["decision"]).all()


def test_check_counts():
    df = score.score(score.clean(reviews()), fake_pipe)
    acc, cm = check.check(df)
    assert 0 <= acc <= 1
    assert cm.sum() == 39


def test_p_positive():
    pos = {"label": "POSITIVE", "score": 0.9}
    assert fails.p_positive(pos) == 0.9
    neg = {"label": "NEGATIVE", "score": 0.9}
    assert abs(fails.p_positive(neg) - 0.1) < 1e-9


def test_head_tail():
    from head_tail import head_tail

    short = list(range(10))
    assert head_tail(short) == short
    out = head_tail(list(range(1000)))
    assert len(out) == 510
    assert out[:128] == list(range(128))
    assert out[-1] == 999
