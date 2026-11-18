"""Two questions that show where the app goes wrong."""
import sys

from ask import answer
from common import INDEX, get_chat, get_embed
from embed import load_index

NOT_IN_DOCS = "Does Harbor Street rent electric scooters?"
LATE = "What is the late fee if I keep a bike overnight?"


def show(question, chunks, matrix, chat, embed):
    reply, hits = answer(question, chunks, matrix, chat, embed)
    print("Q:", question)
    print("A:", reply)
    for n, hit in enumerate(hits, start=1):
        print(f"  [{n}] score {hit['score']:.3f} "
              f"page {hit['page']}")
    print()


if __name__ == "__main__":
    chunks, matrix = load_index(INDEX)
    chat, embed = get_chat(), get_embed()
    questions = sys.argv[1:] or [NOT_IN_DOCS, LATE]
    for question in questions:
        show(question, chunks, matrix, chat, embed)
