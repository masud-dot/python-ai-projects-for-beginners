"""Appendix A, Chapter 11: refuse when retrieval finds nothing."""
import sys

sys.path.insert(0, "vendor/ch11")
from ask import SYSTEM, build_prompt  # noqa: E402
from common import CHAT_MODEL  # noqa: E402
from retrieve import search  # noqa: E402

NO_ANSWER = "I could not find that in the documents."
MIN_SCORE = 0.0  # set this from YOUR scores; see Appendix A


def answer(question, chunks, matrix, chat, embed,
           min_score=MIN_SCORE):
    """Like ask.answer, but skip the model on a weak match."""
    hits = search(question, chunks, matrix, embed)
    if hits[0]["score"] < min_score:
        return NO_ANSWER, hits
    response = chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",
             "content": build_prompt(question, hits)},
        ],
        options={"temperature": 0, "seed": 42},
    )
    return response["message"]["content"], hits
