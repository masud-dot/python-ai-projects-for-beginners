"""Shared settings for the local RAG chapter."""
from pathlib import Path

CHAT_MODEL = "llama3.2:3b"
EMBED_MODEL = "embeddinggemma"
DOCS = Path("docs")
INDEX = Path("index")
CHUNK_WORDS = 60
OVERLAP_WORDS = 10
TOP_K = 4


def get_chat():
    """Return ollama.chat (imported here so tests skip it)."""
    from ollama import chat

    return chat


def get_embed():
    """Return ollama.embed (imported here so tests skip it)."""
    from ollama import embed

    return embed
