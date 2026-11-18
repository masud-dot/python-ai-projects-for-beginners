from collections import deque

from maze import MAZE, draw, find, neighbors


def bfs(maze, start, goal):
    """Return (path, cells_visited), or (None, n) if no path."""
    frontier = deque([start])
    came_from = {start: None}
    visited = 0
    while frontier:
        current = frontier.popleft()
        visited += 1
        if current == goal:
            break
        for nxt in neighbors(maze, current):
            if nxt not in came_from:
                came_from[nxt] = current
                frontier.append(nxt)
    else:
        return None, visited
    path = []
    while current is not None:
        path.append(current)
        current = came_from[current]
    return path[::-1], visited


if __name__ == "__main__":
    start, goal = find(MAZE, "S"), find(MAZE, "G")
    path, visited = bfs(MAZE, start, goal)
    print(draw(MAZE, path))
    print(f"Steps: {len(path) - 1}")
    print(f"Cells visited: {visited}")
