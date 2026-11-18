"""Two failures: a made-up fact and a forgotten first turn."""
from chat import run_turn
from common import get_chat
from history import trim_history
from memory_chat import new_history, send

UNKNOWABLE = (
    "Who won the 1987 Tidewater Regional Spelling Bee, and "
    "what word did they spell last?"
)


def long_chat(chat, budget=300, turns=8):
    """Tell the bot a name, chat on, then ask for the name."""
    history = new_history()
    run_turn(history, "My name is Priya.", chat, budget, False)
    for i in range(turns):
        filler = f"Tell me one fact about the number {i + 2}."
        run_turn(history, filler, chat, budget, False)
    first = history[1]["content"]
    ask = "What is my name?"
    reply, _ = run_turn(history, ask, chat, budget, False)
    return first, reply


if __name__ == "__main__":
    chat = get_chat()
    print("Q:", UNKNOWABLE)
    print("A:", send(new_history(), UNKNOWABLE, chat))
    first, reply = long_chat(chat)
    print("\nOldest message still kept:", first)
    print("Q: What is my name?")
    print("A:", reply)
