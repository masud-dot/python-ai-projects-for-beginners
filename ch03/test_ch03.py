# Chapter 3 tests. Tests that need scikit-learn, matplotlib or the
# California housing download skip themselves when those are
# missing, so the tests run offline once the data is cached.
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


def run(name, cwd):
    return subprocess.run(
        [sys.executable, str(HERE / name)],
        capture_output=True, text=True, check=True, cwd=cwd,
    ).stdout


@pytest.fixture(scope="module")
def housing():
    pytest.importorskip("sklearn")
    pytest.importorskip("matplotlib")
    import common
    try:
        return common.load()
    except Exception as exc:
        pytest.skip(f"housing data unavailable: {exc}")


def test_clean_demo_output(tmp_path):
    out = run("clean_demo.py", tmp_path)
    assert "rows kept by dropna: 3" in out
    assert "[8.0, 7.0, 6.0, 5.0, 4.0]" in out
    assert "[41.0, 21.0, nan, nan, nan]" in out


def test_inject_gaps_counts():
    from gaps import inject_gaps
    cols = ["MedInc", "HouseAge", "AveRooms"]
    df = pd.DataFrame({c: np.ones(20640) for c in cols})
    holey = inject_gaps(df, ["HouseAge", "AveRooms"], 0.02, 42)
    assert holey["MedInc"].isna().sum() == 0
    assert holey["HouseAge"].isna().sum() == 415
    assert holey["AveRooms"].isna().sum() == 421
    assert len(holey.dropna()) == 19811
    assert df.isna().sum().sum() == 0
    assert holey.fillna(holey.median()).isna().sum().sum() == 0


def test_chained_assignment_does_not_change_data():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.warns(Warning):
        df["a"][df["a"] > 1] = 0.0
    assert df["a"].tolist() == [1.0, 2.0, 3.0]
    df.loc[df["a"] > 1, "a"] = 0.0
    assert df["a"].tolist() == [1.0, 0.0, 0.0]


def test_dataset_shape(housing):
    assert housing.shape == (20640, 9)
    assert not housing.isna().any().any()
    assert housing["MedHouseVal"].between(0.1, 5.1).all()


def test_split_is_repeatable(housing):
    import common
    a = common.split(housing)
    b = common.split(housing)
    assert a[0].index.equals(b[0].index)
    assert len(a[0]) == 16512 and len(a[1]) == 4128


def test_scaler_learns_from_train_only(housing):
    import common
    X_train, X_test, _, _ = common.split(housing)
    prep = common.preprocessor(X_train.columns).fit(X_train)
    scaler = prep.named_transformers_["num"][-1]
    assert np.allclose(scaler.mean_, X_train.mean())
    assert not np.allclose(scaler.mean_, housing.drop(
        columns="MedHouseVal").mean())


def test_pipeline_survives_gaps(housing):
    import common
    from gaps import inject_gaps
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    holey = inject_gaps(housing, ["HouseAge"], 0.05, 1)
    X_train, X_test, y_train, _ = common.split(holey)
    pipe = make_pipeline(
        common.preprocessor(X_train.columns), LinearRegression()
    ).fit(X_train, y_train)
    assert not np.isnan(pipe.predict(X_test)).any()


def test_baseline_rmse_matches_std(housing):
    out = run("baseline.py", HERE)
    base = float(out.split("baseline RMSE ")[1].split()[0])
    std = float(out.split("std of test prices ")[1].split()[0])
    assert abs(base - std) < 0.05


def test_models_are_ordered(housing):
    import models
    rows = models.evaluate()
    assert rows["forest"][2] < rows["linear"][2]
    assert rows["linear"][2] < rows["baseline"][2]


def test_forest_cannot_extrapolate(housing):
    import common
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import make_pipeline
    X_train, X_test, y_train, _ = common.split(housing)
    pipe = make_pipeline(
        common.preprocessor(X_train.columns),
        RandomForestRegressor(n_estimators=20, random_state=0),
    ).fit(X_train, y_train)
    row = X_test.iloc[[0]].copy()
    row["MedInc"] = 100.0
    assert pipe.predict(row)[0] <= y_train.max()


@pytest.mark.parametrize("name", [
    "load_data.py", "clean_housing.py", "split_pipeline.py",
    "curves.py", "checks.py", "failures.py",
])
def test_scripts_run(housing, tmp_path, name):
    assert run(name, tmp_path).strip()


def test_tune_output(housing, tmp_path):
    pytest.importorskip("sklearn")
    out = run("tune.py", tmp_path)
    assert "min_samples_leaf 1: cv RMSE 0.511" in out
    assert "ridge test RMSE 0.745" in out
    assert "Latitude    -0.897" in out
