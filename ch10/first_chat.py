"""Send one prompt to the local model and print the reply."""
import sys

from common import MODEL, get_chat


def ask(prompt, chat=None):
    """Send one user message and return the reply text."""
    chat = chat or get_chat()
    response = chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Why is the sky blue?"
    print(ask(question))
