import heapq

from compare import DETOUR
from maze import MAZE, find, neighbors


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def best_first(maze, start, goal, g_w, h_w):
    """Sort by g_w * cost + h_w * estimate."""
    count = 0
    frontier = [(0, count, start)]
    cost = {start: 0}
    came_from = {start: None}
    pops = 0
    popped = set()
    repeats = 0
    while frontier:
        _, _, current = heapq.heappop(frontier)
        pops += 1
        repeats += current in popped
        popped.add(current)
        if current == goal:
            break
        for nxt in neighbors(maze, current):
            new_cost = cost[current] + 1
            if nxt not in cost or new_cost < cost[nxt]:
                cost[nxt] = new_cost
                came_from[nxt] = current
                count += 1
                h = manhattan(nxt, goal)
                priority = g_w * new_cost + h_w * h
                heapq.heappush(frontier, (priority, count, nxt))
    steps = 0
    while came_from[current] is not None:
        current = came_from[current]
        steps += 1
    return steps, pops, repeats


RULES = [
    ("A*", 1, 1),
    ("greedy", 0, 1),
    ("2x estimate", 1, 2),
]

if __name__ == "__main__":
    for name, maze in [("main", MAZE), ("detour", DETOUR)]:
        start, goal = find(maze, "S"), find(maze, "G")
        for label, g_w, h_w in RULES:
            result = best_first(maze, start, goal, g_w, h_w)
            steps, pops, again = result
            print(f"{name:7}{label:12}{steps:3} steps,"
                  f" {pops:3} pops, {again} repeats")
