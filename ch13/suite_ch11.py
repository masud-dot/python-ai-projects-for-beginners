import sys

sys.path.insert(0, "vendor/ch11")
import helpers_ch11 as t11  # noqa: E402
import threshold_ask as t  # noqa: E402
from embed import embed_texts  # noqa: E402


def test_cutoff_skips_the_model():
    chunks = [{"id": 0, "source": "d.pdf", "page": 1,
               "text": "The day pass costs 6 dollars."}]
    matrix = embed_texts([c["text"] for c in chunks], t11.fake_embed)
    chat = t11.FakeChat()
    reply, _ = t.answer("day pass cost", chunks, matrix, chat,
                        t11.fake_embed, min_score=0.5)
    assert "6 dollars" in reply and len(chat.calls) == 1
    reply, hits = t.answer("zebra telescope", chunks, matrix, chat,
                           t11.fake_embed, min_score=0.5)
    assert reply == t.NO_ANSWER and len(chat.calls) == 1
