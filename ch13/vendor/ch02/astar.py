import heapq

from maze import MAZE, find, neighbors
from bfs import bfs


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(maze, start, goal):
    """Return (path, cells_visited), or (None, n) if no path."""
    count = 0  # tie-breaker so the heap never compares cells
    frontier = [(manhattan(start, goal), count, start)]
    came_from = {start: None}
    cost = {start: 0}
    visited = 0
    while frontier:
        _, _, current = heapq.heappop(frontier)
        visited += 1
        if current == goal:
            break
        for nxt in neighbors(maze, current):
            new_cost = cost[current] + 1
            if nxt not in cost or new_cost < cost[nxt]:
                cost[nxt] = new_cost
                came_from[nxt] = current
                count += 1
                priority = new_cost + manhattan(nxt, goal)
                heapq.heappush(frontier, (priority, count, nxt))
    else:
        return None, visited
    path = []
    while current is not None:
        path.append(current)
        current = came_from[current]
    return path[::-1], visited


if __name__ == "__main__":
    start, goal = find(MAZE, "S"), find(MAZE, "G")
    b_path, b_seen = bfs(MAZE, start, goal)
    a_path, a_seen = astar(MAZE, start, goal)
    b_steps, a_steps = len(b_path) - 1, len(a_path) - 1
    print(f"BFS:  {b_steps} steps, {b_seen} cells visited")
    print(f"A*:   {a_steps} steps, {a_seen} cells visited")
