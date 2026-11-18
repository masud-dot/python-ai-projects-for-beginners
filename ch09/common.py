"""Settings and loader shared by the chapter 9 scripts."""
from pathlib import Path

MODEL = (
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)
HERE = Path(__file__).parent
RESULTS = HERE / "results"


def load_pipe():
    """Load the named model on the CPU (downloads it once)."""
    from transformers import pipeline

    return pipeline(
        "text-classification", model=MODEL, device="cpu"
    )
