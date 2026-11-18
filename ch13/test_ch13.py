"""Run each Appendix A suite in its own process.

Chapters 8, 10 and 11 each have a common.py, so one process
cannot import them together.
"""
import subprocess
import sys

import pytest

SUITES = ["suite_ch02_ch07.py", "suite_ch08.py", "suite_ch09.py",
          "suite_ch10.py", "suite_ch11.py"]


@pytest.mark.parametrize("suite", SUITES)
def test_suite(suite):
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p",
         "no:cacheprovider", suite],
        capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr


def test_manhattan_trap_maze(monkeypatch):
    """The 5x5 maze in the Chapter 2 answer: 6 against 5 steps."""
    here = __import__("os").path.dirname(__file__)
    monkeypatch.chdir(here)
    monkeypatch.syspath_prepend(here)
    import diagonal as d
    maze = ["..#..", ".#.#.", "#....", "..#..", "....."]
    s, g = (0, 0), (4, 4)
    assert d.bfs8(maze, s, g)[0] == 5
    assert d.astar8(maze, s, g, d.manhattan)[0] == 6
    assert d.astar8(maze, s, g, d.chebyshev)[0] == 5
