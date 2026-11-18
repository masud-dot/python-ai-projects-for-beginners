"""Run the sentiment model on five sentences."""
from common import MODEL, load_pipe

SENTENCES = [
    "This restaurant is awesome",
    "The delivery was late and the box was crushed.",
    "It works. I have no strong feelings about it.",
    "I would happily buy this again.",
    "Worst purchase I have made all year.",
]

if __name__ == "__main__":
    pipe = load_pipe()
    print("model:", MODEL)
    for text, res in zip(SENTENCES, pipe(SENTENCES)):
        print(f"{res['label']:<8} {res['score']:.4f}  {text}")
