"""Create the standard project folder for a new project."""
import sys
from pathlib import Path

FOLDERS = ["data", "src", "results"]
GITIGNORE = ".venv/\n__pycache__/\ndata/\n"


def make_project(name, root="."):
    """Build <root>/<name> with data, src, results folders."""
    project = Path(root) / name
    for folder in FOLDERS:
        (project / folder).mkdir(parents=True, exist_ok=True)
    for filename, text in [
        ("requirements.txt", ""),
        (".gitignore", GITIGNORE),
        ("README.md", f"# {name}\n"),
    ]:
        path = project / filename
        if not path.exists():
            path.write_text(text, encoding="utf-8")
    return project


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python make_project.py PROJECT_NAME")
    created = make_project(sys.argv[1])
    for path in sorted(created.rglob("*")):
        print(path)
