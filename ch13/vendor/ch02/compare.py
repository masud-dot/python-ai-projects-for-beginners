from astar import astar
from bfs import bfs
from maze import find

OPEN = [
    ".........",
    ".........",
    ".........",
    "S.....G..",
    ".........",
    ".........",
    ".........",
]

DETOUR = [
    ".........",
    "...#.....",
    "...#.....",
    "S..#..G..",
    "...#.....",
    "...#.....",
    ".........",
]


def compare(name, maze):
    start, goal = find(maze, "S"), find(maze, "G")
    b_path, b_seen = bfs(maze, start, goal)
    a_path, a_seen = astar(maze, start, goal)
    print(f"{name}: {len(b_path) - 1} steps")
    print(f"  BFS visited {b_seen}, A* visited {a_seen}")


if __name__ == "__main__":
    compare("open", OPEN)
    compare("detour", DETOUR)
