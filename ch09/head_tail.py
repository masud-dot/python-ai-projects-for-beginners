def head_tail(items, head=128, tail=382):
    """Keep the first `head` and last `tail` items of a list.

    Token ids work well: 128 + 382 + 2 special tokens = 512.
    A list that already fits is returned unchanged.
    """
    if len(items) <= head + tail:
        return list(items)
    return list(items[:head]) + list(items[-tail:])
