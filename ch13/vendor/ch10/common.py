"""Shared settings for the chatbot chapter."""
MODEL = "llama3.2:3b"
SYSTEM = (
    "You are Ollie, a concise, friendly assistant. Answer in "
    "at most three sentences. If you do not know a fact, say "
    "so instead of guessing."
)


def get_chat():
    """Return ollama.chat (imported here so tests skip it)."""
    from ollama import chat

    return chat
