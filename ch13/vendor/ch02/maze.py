MAZE = [
    "S.........#.....",
    "..........#.....",
    "..........#.....",
    "....#####.#.....",
    "....#...#.#.....",
    "....#.#.#.#.....",
    "......#.........",
    ".######.#######.",
    "......#........G",
]


def find(maze, mark):
    """Return the (row, col) of the first cell equal to mark."""
    for r, line in enumerate(maze):
        if mark in line:
            return (r, line.index(mark))
    raise ValueError(f"no {mark!r} in maze")


def neighbors(maze, cell):
    """Yield the open cells one step up, down, left or right."""
    r, c = cell
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < len(maze) and 0 <= nc < len(maze[nr]):
            if maze[nr][nc] != "#":
                yield (nr, nc)


def draw(maze, path):
    """Return the maze as text with the path marked by stars."""
    on_path = set(path)
    rows = []
    for r, line in enumerate(maze):
        row = ""
        for c, ch in enumerate(line):
            if (r, c) in on_path and ch == ".":
                ch = "*"
            row += ch
        rows.append(row)
    return "\n".join(rows)
