import subprocess
import sys
from pathlib import Path

import smoke_test
from make_project import make_project
from seed_everything import draw_numbers

HERE = Path(__file__).parent


def run(name):
    return subprocess.run(
        [sys.executable, str(HERE / name)],
        capture_output=True, text=True, check=True,
    ).stdout


def test_script_and_rerunnable_match():
    assert run("ten_lines.py") == run("ten_lines_rerunnable.py")


def test_make_project(tmp_path):
    project = make_project("demo", tmp_path)
    for folder in ["data", "src", "results"]:
        assert (project / folder).is_dir()
    assert (project / "requirements.txt").exists()
    make_project("demo", tmp_path)  # safe to rerun


def test_seed_is_repeatable():
    assert draw_numbers(42) == draw_numbers(42)
    assert draw_numbers(42) != draw_numbers(7)


def test_smoke_rows():
    rows = smoke_test.check_all()
    assert len(rows) == len(smoke_test.PACKAGES)
    names = {name for name, _, _ in rows}
    assert "numpy" in names


def test_requirements_pinned():
    lines = (HERE / "requirements.txt").read_text().split()
    assert "torch==2.15.0" in lines
    assert "numpy==2.5.4" in lines


def test_tensor_check_reports():
    ok, message = smoke_test.tensor_check()
    assert isinstance(ok, bool)
    assert message


def test_unseeded_draws_differ():
    import numpy as np

    def draw():
        rng = np.random.default_rng()
        return round(float(rng.random()), 4)

    assert draw() != draw()
