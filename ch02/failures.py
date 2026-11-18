from astar import astar
from bfs import bfs
from descent import fit, loss
from maze import find

WALLED = [
    "S.#..",
    "..#..",
    "..#.G",
]


def show_unreachable():
    start, goal = find(WALLED, "S"), find(WALLED, "G")
    for search in (bfs, astar):
        path, visited = search(WALLED, start, goal)
        name = search.__name__
        print(f"{name}: path={path}, visited={visited}")


def show_divergence():
    fit(rate=0.05, steps=6, report=1)


def rate_limit(low=0.01, high=0.05):
    """Bisect for the largest rate whose loss still falls."""
    for _ in range(12):
        mid = (low + high) / 2
        w, b = fit(rate=mid, steps=250, report=0)
        if loss(w, b) < loss(0.0, 0.0):
            low = mid
        else:
            high = mid
    print(f"largest safe rate is near {low:.4f}")


if __name__ == "__main__":
    show_unreachable()
    show_divergence()
    rate_limit()
