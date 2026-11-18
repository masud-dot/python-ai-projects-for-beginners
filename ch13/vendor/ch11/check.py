"""Ten questions with known pages: does retrieval find them?"""
from common import INDEX, TOP_K, get_embed
from embed import load_index
from retrieve import search

SOURCE = "harbor_bikes.pdf"
QUESTIONS = [
    ("How much does a day pass cost?", 1),
    ("How long can a ride last on a monthly membership?", 1),
    ("What discount do students get?", 1),
    ("How old must a rider be?", 2),
    ("Can I ride on the sidewalk downtown?", 2),
    ("How heavy can the basket load be?", 2),
    ("What should I do if the dock light is red?", 3),
    ("What do I pay if a bike is not returned in 24 hours?", 3),
    ("When is the help line open?", 4),
    ("Can I get a refund on an annual membership?", 4),
]


def check_question(question, page, chunks, matrix, embed, k):
    """True when a top-k chunk comes from the expected page."""
    hits = search(question, chunks, matrix, embed, k)
    return any(
        (h["source"], h["page"]) == (SOURCE, page)
        for h in hits
    )


def run_checks(chunks, matrix, embed, k=TOP_K):
    """Return [(question, page, found)] for every question."""
    return [
        (q, page, check_question(
            q, page, chunks, matrix, embed, k))
        for q, page in QUESTIONS
    ]


if __name__ == "__main__":
    chunks, matrix = load_index(INDEX)
    results = run_checks(chunks, matrix, get_embed())
    for q, page, found in results:
        print(f"{'PASS' if found else 'FAIL'}  p.{page}  {q}")
    passed = sum(found for _, _, found in results)
    print(f"{passed} of {len(results)} found the right page")
