"""Keep a chat history inside the model's context window."""


def estimate_tokens(text):
    """Rough guess: one token is about four characters."""
    return len(text) // 4 + 1


def history_tokens(messages):
    """Estimated tokens for a whole list of messages."""
    return sum(estimate_tokens(m["content"]) for m in messages)


def trim_history(messages, budget=3000):
    """Drop the oldest user/assistant pairs until it fits.

    The system prompt (first message) is always kept, and so
    is the newest message. Returns a new list.
    """
    system = [m for m in messages[:1] if m["role"] == "system"]
    rest = list(messages[len(system):])
    while len(rest) > 1:
        if history_tokens(system + rest) <= budget:
            break
        rest = rest[2:]
    return system + rest
