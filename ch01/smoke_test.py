"""Import every library the book uses and print versions."""
import importlib
import platform
import sys
from importlib import metadata

PACKAGES = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("scikit-learn", "sklearn"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("transformers", "transformers"),
    ("faster-whisper", "faster_whisper"),
    ("sentence-transformers", "sentence_transformers"),
    ("ollama", "ollama"),
    ("pypdf", "pypdf"),
]


def check_all():
    """Return (name, version or None, error or None) rows."""
    rows = []
    for dist, module in PACKAGES:
        try:
            importlib.import_module(module)
            rows.append((dist, metadata.version(dist), None))
        except Exception as error:  # report, don't crash
            rows.append((dist, None, repr(error)))
    return rows


def tensor_check():
    """Run a 2x2 multiply; return (ok, message)."""
    try:
        import torch
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        total = float((a @ a).sum())
        if total != 54.0:
            return False, f"wrong result {total}"
        gpu = torch.cuda.is_available()
        return True, f"GPU available: {gpu}"
    except Exception as error:  # any broken-install error
        return False, repr(error)


if __name__ == "__main__":
    print(f"Python {platform.python_version()}")
    failed = 0
    for name, version, error in check_all():
        if error:
            failed += 1
            print(f"FAIL {name}: {error}")
        else:
            print(f"ok   {name} {version}")
    ok, message = tensor_check()
    if ok:
        print("ok   tensor multiply")
        print(message)
    else:
        failed += 1
        print(f"FAIL tensor multiply: {message}")
    sys.exit(1 if failed else 0)
