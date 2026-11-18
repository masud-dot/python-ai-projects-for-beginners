"""Appendix A, Chapter 10: save and load the chat history."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "vendor/ch10")
from common import get_chat  # noqa: E402
from chat import BUDGET, run_turn  # noqa: E402
from history import trim_history  # noqa: E402
from memory_chat import SYSTEM, new_history  # noqa: E402

PATH = Path("history.json")


def save_history(history, path=PATH):
    Path(path).write_text(json.dumps(history, indent=2),
                          encoding="utf-8")


def valid(history):
    return (isinstance(history, list) and history
            and all(isinstance(m, dict)
                    and m.get("role") in {"system", "user",
                                          "assistant"}
                    and isinstance(m.get("content"), str)
                    for m in history))


def load_history(path=PATH, system=SYSTEM):
    """Return the saved history, or a new one if unusable."""
    try:
        history = json.loads(
            Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return new_history(system)
    if not valid(history):
        return new_history(system)
    if history[0]["role"] == "system":
        history[0]["content"] = system  # use today's prompt
    else:
        first = {"role": "system", "content": system}
        history.insert(0, first)
    return trim_history(history, BUDGET)


def main():
    chat = get_chat()
    history = load_history()
    print("Type 'quit' to stop.")
    while True:
        text = input("You: ").strip()
        if text == "quit":
            break
        if text:
            print("Bot: ", end="")
            run_turn(history, text, chat)
    save_history(history)


if __name__ == "__main__":
    main()
