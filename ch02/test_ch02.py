import subprocess
import sys
from pathlib import Path

import checks
import failures
from astar import astar
from bfs import bfs
from descent import fit, loss
from maze import MAZE, find

HERE = Path(__file__).parent


def run(name):
    return subprocess.run(
        [sys.executable, str(HERE / name)],
        capture_output=True, text=True, check=True,
    ).stdout


def test_bfs_script_output():
    out = run("bfs.py")
    assert "Steps: 23" in out
    assert "Cells visited: 100" in out


def test_astar_visits_fewer_cells_same_length():
    start, goal = find(MAZE, "S"), find(MAZE, "G")
    b_path, b_seen = bfs(MAZE, start, goal)
    a_path, a_seen = astar(MAZE, start, goal)
    assert len(a_path) == len(b_path)
    assert a_seen < b_seen


def test_astar_script_output():
    out = run("astar.py")
    assert "BFS:  23 steps, 100 cells visited" in out
    assert "A*:   23 steps, 67 cells visited" in out


def test_descent_loss_falls():
    assert loss(0.0, 0.0) > loss(*fit(steps=200, report=0))


def test_descent_script_output():
    out = run("descent.py")
    assert "w = 1.9962, b = 1.0666" in out


def test_checks_pass(capsys):
    checks.check_maze()
    checks.check_line()
    assert "line checks passed" in capsys.readouterr().out


def test_unreachable_goal(capsys):
    failures.show_unreachable()
    out = capsys.readouterr().out
    assert "bfs: path=None" in out and "astar: path=None" in out


def test_divergence(capsys):
    failures.show_divergence()
    losses = [
        float(line.split()[-1])
        for line in capsys.readouterr().out.splitlines()
    ]
    assert losses == sorted(losses) and losses[-1] > 1e5


def test_one_number_script():
    out = run("one_number.py").splitlines()
    assert out[0] == "step 0  x = 0.0000  f(x) = 9.0000"
    assert out[-1] == "step 5  x = 2.9062  f(x) = 0.0088"


def test_compare_script():
    out = run("compare.py")
    assert "BFS visited 37, A* visited 7" in out
    assert "BFS visited 50, A* visited 44" in out


def test_check_path_rejects_a_jump():
    import pytest
    with pytest.raises(AssertionError):
        checks.check_path(checks.TINY, [(0, 0), (2, 2)])


def test_heuristics_script_output():
    out = run("heuristics.py").splitlines()
    assert "23 steps,  67 pops, 0 repeats" in out[0]
    assert "27 steps,  35 pops, 0 repeats" in out[1]
    assert "27 steps,  43 pops, 2 repeats" in out[2]
    assert "12 steps,  21 pops, 0 repeats" in out[4]


def test_rate_limit(capsys):
    failures.rate_limit()
    out = capsys.readouterr().out
    assert "largest safe rate is near 0.0342" in out


def test_do_it_counts():
    import bfs as bfs_mod
    import maze
    from collections import deque
    start, goal = find(MAZE, "S"), find(MAZE, "G")
    order = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def reordered(m, cell):
        r, c = cell
        for dr, dc in order:
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(m) and 0 <= nc < len(m[nr]):
                if m[nr][nc] != "#":
                    yield (nr, nc)

    real = bfs_mod.neighbors
    bfs_mod.neighbors = reordered
    try:
        path, seen = bfs_mod.bfs(MAZE, start, goal)
    finally:
        bfs_mod.neighbors = real
    assert (len(path) - 1, seen) == (23, 94)

    frontier, came, seen = deque([start]), {start: None}, 0
    while frontier:
        cur = frontier.pop()
        seen += 1
        if cur == goal:
            break
        for nxt in maze.neighbors(MAZE, cur):
            if nxt not in came:
                came[nxt] = cur
                frontier.append(nxt)
    steps = 0
    while came[cur] is not None:
        cur, steps = came[cur], steps + 1
    assert (steps, seen) == (45, 46)

    w, b = fit(rate=0.001, steps=1000, report=0)
    assert round(loss(w, b), 2) == 0.07


def test_long_run_matches_closed_form():
    w, b = fit(steps=50000, report=0)
    w_exact, b_exact = checks.closed_form(*checks_data())
    assert abs(w - w_exact) < 1e-4 and abs(b - b_exact) < 1e-4


def checks_data():
    from descent import XS, YS
    return XS, YS


def test_slope_nudge_checks():
    from descent import loss as lo
    assert abs((lo(0.001, 0) - lo(0, 0)) + 0.1234) < 1e-3
    assert abs(((0.01 - 3) ** 2 - 9) + 0.0599) < 1e-4
