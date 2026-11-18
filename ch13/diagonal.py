"""Appendix A, Chapter 2: diagonal moves."""
import heapq
import sys
from collections import deque

sys.path.insert(0, "vendor/ch02")
from maze import MAZE, find  # noqa: E402

STEPS = [(-1, 0), (1, 0), (0, -1), (0, 1),
         (-1, -1), (-1, 1), (1, -1), (1, 1)]


def neighbors8(maze, cell):
    r, c = cell
    for dr, dc in STEPS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < len(maze) and 0 <= nc < len(maze[nr]):
            if maze[nr][nc] != "#":
                yield (nr, nc)


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def chebyshev(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def bfs8(maze, start, goal):
    frontier, seen, visited = deque([(start, 0)]), {start}, 0
    while frontier:
        cell, dist = frontier.popleft()
        visited += 1
        if cell == goal:
            return dist, visited
        for nxt in neighbors8(maze, cell):
            if nxt not in seen:
                seen.add(nxt)
                frontier.append((nxt, dist + 1))
    return None, visited


def astar8(maze, start, goal, h):
    frontier, cost, visited = [(h(start, goal), 0, start)], {start: 0}, 0
    count = 0
    while frontier:
        _, _, cell = heapq.heappop(frontier)
        visited += 1
        if cell == goal:
            return cost[cell], visited
        for nxt in neighbors8(maze, cell):
            new = cost[cell] + 1
            if nxt not in cost or new < cost[nxt]:
                cost[nxt] = new
                count += 1
                heapq.heappush(frontier, (new + h(nxt, goal), count, nxt))
    return None, visited


if __name__ == "__main__":
    start, goal = find(MAZE, "S"), find(MAZE, "G")
    print("BFS:", *bfs8(MAZE, start, goal), "(steps, visited)")
    for name, h in [("manhattan", manhattan), ("chebyshev", chebyshev)]:
        print(f"A* {name}:", *astar8(MAZE, start, goal, h))
    print("3 rows, 3 cols apart: manhattan", manhattan((0, 0), (3, 3)),
          "true", bfs8(["...."] * 4, (0, 0), (3, 3))[0])
