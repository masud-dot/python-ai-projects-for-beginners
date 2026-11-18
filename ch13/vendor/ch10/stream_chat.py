"""Print the reply token by token and report the speed."""
from common import MODEL


def tokens_per_second(chunk):
    """Speed from the last chunk: eval_count / eval_duration."""
    return chunk["eval_count"] / chunk["eval_duration"] * 1e9


def stream_reply(history, chat, options=None, show=True):
    """Stream one reply. Return (text, tokens per second)."""
    parts = []
    speed = None
    stream = chat(
        model=MODEL,
        messages=history,
        options=options,
        stream=True,
    )
    for chunk in stream:
        piece = chunk["message"]["content"]
        parts.append(piece)
        if show:
            print(piece, end="", flush=True)
        if chunk.get("done"):
            speed = tokens_per_second(chunk)
    if show:
        print()
    return "".join(parts), speed
