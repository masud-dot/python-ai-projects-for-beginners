"""The finished chatbot: memory, streaming and trimming."""
from common import get_chat
from history import trim_history
from memory_chat import new_history
from stream_chat import stream_reply

BUDGET = 3000  # estimated tokens; the default window is 4096


def run_turn(history, text, chat, budget=BUDGET, show=True):
    """Add the user text, trim, stream the reply, store it."""
    history.append({"role": "user", "content": text})
    history[:] = trim_history(history, budget)
    reply, speed = stream_reply(history, chat, show=show)
    history.append({"role": "assistant", "content": reply})
    return reply, speed


def main():
    chat = get_chat()
    history = new_history()
    print("Type 'quit' to stop.")
    while True:
        text = input("You: ").strip()
        if text == "quit":
            break
        if not text:
            continue
        print("Bot: ", end="")
        _, speed = run_turn(history, text, chat)
        if speed:
            print(f"[{speed:.1f} tokens/s]")


if __name__ == "__main__":
    main()
