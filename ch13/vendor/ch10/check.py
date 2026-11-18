"""Scripted conversation: memory and system prompt."""
from common import get_chat
from memory_chat import new_history, send

OPTIONS = {"temperature": 0, "seed": 42}
TAG = "-- Ollie"


def check_memory(chat):
    history = new_history()
    send(history, "My cat is called Biscuit.", chat, OPTIONS)
    ask = "What is my cat called?"
    reply = send(history, ask, chat, OPTIONS)
    return "biscuit" in reply.lower()


def check_system_prompt(chat):
    system = f"Always end every reply with the text {TAG}"
    history = new_history(system)
    ask = "Name a primary color."
    reply = send(history, ask, chat, OPTIONS)
    return reply.strip().endswith(TAG)


def check_no_memory(chat):
    """Control: a fresh history must not know the cat's name."""
    history = new_history()
    ask = "What is my cat called?"
    reply = send(history, ask, chat, OPTIONS)
    return "biscuit" not in reply.lower()


if __name__ == "__main__":
    chat = get_chat()
    checks = (
        check_memory,
        check_system_prompt,
        check_no_memory,
    )
    for fn in checks:
        ok = "PASS" if fn(chat) else "FAIL"
        print(f"{ok}  {fn.__name__}")
