from astar import astar
from bfs import bfs
from descent import XS, YS, fit, loss
from maze import MAZE, find

TINY = [
    "S.#",
    ".##",
    "..G",
]


def closed_form(xs, ys):
    """Least-squares line through the points, no loop needed."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    w = sxy / sxx
    return w, my - w * mx


def check_path(maze, path):
    """Every step must be one move onto an open cell."""
    assert maze[path[0][0]][path[0][1]] == "S"
    assert maze[path[-1][0]][path[-1][1]] == "G"
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        assert abs(r1 - r2) + abs(c1 - c2) == 1, "not one step"
        assert maze[r2][c2] != "#", "path walks through a wall"


def check_maze():
    start, goal = find(TINY, "S"), find(TINY, "G")
    for search in (bfs, astar):
        path, _ = search(TINY, start, goal)
        assert len(path) - 1 == 4, "hand count says 4 steps"
        check_path(TINY, path)
    start, goal = find(MAZE, "S"), find(MAZE, "G")
    b_path, _ = bfs(MAZE, start, goal)
    a_path, _ = astar(MAZE, start, goal)
    assert len(b_path) == len(a_path), "lengths must match"
    check_path(MAZE, b_path)
    check_path(MAZE, a_path)
    print("maze checks passed")


def check_line():
    w, b = fit(report=0)
    w_exact, b_exact = closed_form(XS, YS)
    print(f"descent:     w = {w:.4f}, b = {b:.4f}")
    print(f"closed form: w = {w_exact:.4f}, b = {b_exact:.4f}")
    assert abs(w - w_exact) < 0.01 and abs(b - b_exact) < 0.01
    assert loss(w, b) <= loss(w_exact, b_exact) + 1e-3
    print("line checks passed")


if __name__ == "__main__":
    check_maze()
    check_line()
