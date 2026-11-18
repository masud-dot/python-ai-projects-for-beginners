"""Appendix A, Chapter 2: weighted cells."""
import heapq
import sys
from collections import deque

sys.path.insert(0, "vendor/ch02")
from maze import find, neighbors  # noqa: E402

COSTLY = [
    "S.9.G",
    ".###.",
    ".....",
]
# Short way: along the top row through the 9: 4 steps, cost 12.
# Detour: down, across the bottom row and up: 8 steps, cost 8.


def enter_cost(maze, cell):
    ch = maze[cell[0]][cell[1]]
    return int(ch) if ch.isdigit() else 1


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def bfs_path(maze, start, goal):
    came, frontier = {start: None}, deque([start])
    while frontier:
        cur = frontier.popleft()
        if cur == goal:
            break
        for nxt in neighbors(maze, cur):
            if nxt not in came:
                came[nxt] = cur
                frontier.append(nxt)
    return came, goal


def astar_weighted(maze, start, goal):
    came, cost, count = {start: None}, {start: 0}, 0
    frontier = [(manhattan(start, goal), 0, start)]
    while frontier:
        _, _, cur = heapq.heappop(frontier)
        if cur == goal:
            break
        for nxt in neighbors(maze, cur):
            new_cost = cost[cur] + enter_cost(maze, nxt)
            if nxt not in cost or new_cost < cost[nxt]:
                cost[nxt] = new_cost
                came[nxt] = cur
                count += 1
                guess = new_cost + manhattan(nxt, goal)
                heapq.heappush(frontier, (guess, count, nxt))
    return came, cost[goal]


def path_of(came, goal):
    path = []
    while goal is not None:
        path.append(goal)
        goal = came[goal]
    return path[::-1]


def total(maze, path):
    return sum(enter_cost(maze, c) for c in path[1:])


if __name__ == "__main__":
    start, goal = find(COSTLY, "S"), find(COSTLY, "G")
    came, _ = bfs_path(COSTLY, start, goal)
    b = path_of(came, goal)
    came, cost = astar_weighted(COSTLY, start, goal)
    a = path_of(came, goal)
    print("BFS path:", len(b) - 1, "steps, total cost", total(COSTLY, b))
    print("A*  path:", len(a) - 1, "steps, total cost", cost)
