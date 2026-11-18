"""Probe the model with sarcasm and name-based differences."""
from common import load_pipe

SARCASM = [
    "Oh great, another update that deletes my settings.",
    "Wow, thanks for the two-hour wait. Truly a treat.",
    "I can't say I'm unhappy with it.",
]
TEMPLATE = "This film was filmed in {}."
PLACES = ["France", "Afghanistan", "Canada", "Nigeria"]


def p_positive(res):
    """Positive-side confidence from one pipeline result."""
    s = res["score"]
    return s if res["label"] == "POSITIVE" else 1 - s


if __name__ == "__main__":
    pipe = load_pipe()
    print("Sarcasm and negation:")
    for text, res in zip(SARCASM, pipe(SARCASM)):
        print(f"{res['label']:<8} {res['score']:.3f}  {text}")
    print("\nSame sentence, different place:")
    texts = [TEMPLATE.format(p) for p in PLACES]
    for text, res in zip(texts, pipe(texts)):
        print(f"{p_positive(res):.3f}  {text}")
