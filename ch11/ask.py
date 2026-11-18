"""Ask questions about your documents, with cited sources."""
from common import CHAT_MODEL, INDEX, get_chat, get_embed
from embed import load_index
from retrieve import search

SYSTEM = (
    "You answer questions using only the numbered passages "
    "in the user's message. Cite the passage numbers you "
    "used, like [1]. If the passages do not contain the "
    "answer, reply exactly: I could not find that in the "
    "documents."
)


def build_prompt(question, hits):
    """Number the passages and add the question."""
    lines = ["Passages:"]
    for number, hit in enumerate(hits, start=1):
        lines.append(f"[{number}] {hit['text']}")
    lines.append("")
    lines.append(f"Question: {question}")
    return "\n".join(lines)


def format_sources(hits):
    """List the passages the model was shown."""
    return "\n".join(
        f"[{n}] {h['source']}, page {h['page']}"
        for n, h in enumerate(hits, start=1)
    )


def answer(question, chunks, matrix, chat, embed):
    """Retrieve, ask the model, return (reply, hits)."""
    hits = search(question, chunks, matrix, embed)
    response = chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",
             "content": build_prompt(question, hits)},
        ],
        options={"temperature": 0, "seed": 42},
    )
    return response["message"]["content"], hits


def main():
    chunks, matrix = load_index(INDEX)
    chat, embed = get_chat(), get_embed()
    print("Ask about your documents. Type 'quit' to stop.")
    while True:
        question = input("Question: ").strip()
        if question == "quit":
            break
        if not question:
            continue
        reply, hits = answer(
            question, chunks, matrix, chat, embed
        )
        print("\n" + reply)
        print("\nSources shown to the model:")
        print(format_sources(hits) + "\n")


if __name__ == "__main__":
    main()
