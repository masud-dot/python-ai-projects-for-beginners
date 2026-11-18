"""A chatbot that remembers: it resends the whole history."""
from common import MODEL, SYSTEM, get_chat


def new_history(system=SYSTEM):
    """Start a conversation with only the system prompt."""
    return [{"role": "system", "content": system}]


def send(history, text, chat, options=None):
    """Add the user text, ask, store and return the reply."""
    history.append({"role": "user", "content": text})
    response = chat(
        model=MODEL, messages=history, options=options
    )
    reply = response["message"]["content"]
    history.append({"role": "assistant", "content": reply})
    return reply


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
        print("Bot:", send(history, text, chat))


if __name__ == "__main__":
    main()
