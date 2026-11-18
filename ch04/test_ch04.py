# Chapter 4 tests. Tests that need scikit-learn, matplotlib or
# the MNIST download skip themselves when those are missing.
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


@pytest.fixture(scope="module")
def mnist():
    pytest.importorskip("sklearn")
    pytest.importorskip("matplotlib")
    import common
    try:
        return common.load()
    except Exception as exc:
        pytest.skip(f"MNIST unavailable: {exc}")


def test_shapes_and_scaling(mnist):
    X, y = mnist
    assert X.shape == (70_000, 784)
    assert y.dtype.kind == "i"
    assert X.min() == 0.0 and X.max() == 1.0


def test_official_split(mnist):
    import common
    X_train, X_test, y_train, y_test = common.split(*mnist)
    assert len(X_train) == 60_000 and len(X_test) == 10_000
    assert len(y_train) == 60_000 and len(y_test) == 10_000


def test_class_counts(mnist):
    import common
    *_, y_train, y_test = common.split(*mnist)
    assert np.bincount(y_test).sum() == 10_000
    assert np.bincount(y_train)[5] == 5_421
    assert np.bincount(y_test)[5] == 892


def test_cache_matches_download(mnist):
    import common
    assert common.CACHE.exists()
    saved = np.load(common.CACHE)
    assert saved["X"].dtype == np.uint8
    assert saved["X"].shape == (70_000, 784)


def test_string_label_pitfall():
    labels = np.array(["5", "0"], dtype=object)
    assert not (labels == 5).any()
    assert (labels == "5").sum() == 1


def test_checks_script_passes(mnist, tmp_path):
    out = subprocess.run(
        [sys.executable, str(HERE / "checks.py")],
        capture_output=True, text=True, check=True,
        cwd=tmp_path,
    ).stdout
    assert "all checks passed" in out
