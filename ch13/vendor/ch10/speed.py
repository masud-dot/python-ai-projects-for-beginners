"""Measure tokens per second for one streamed reply."""
from common import MODEL, get_chat
from memory_chat import new_history
from stream_chat import stream_reply


def measure(chat, question="Explain what a CPU does."):
    """Stream one reply and return its tokens per second."""
    history = new_history()
    history.append({"role": "user", "content": question})
    _, speed = stream_reply(history, chat)
    return speed


if __name__ == "__main__":
    speed = measure(get_chat())
    print(f"{MODEL}: {speed:.1f} tokens/s")
