# Chapter 11 tests. They run offline: a hashing "embedder"
# stands in for the Ollama embedding model and a fake chat
# function stands in for the language model. The real models
# need Ollama, so they are not run here.
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import ask  # noqa: E402
import check  # noqa: E402
import chunker  # noqa: E402
import compare_sizes  # noqa: E402
import embed as emb  # noqa: E402
import index_docs  # noqa: E402
import inspect_index  # noqa: E402
import make_sample  # noqa: E402
import pdf_text  # noqa: E402
import retrieve  # noqa: E402

DIM = 256
STOP = {"the", "a", "an", "of", "is", "to", "in", "on", "if",
        "do", "i", "can", "be", "for", "and", "what", "how"}


def fake_embed(model, input):
    """Hash each word into one of 256 slots (bag of words)."""
    rows = []
    for text in input:
        row = [0.0] * DIM
        for word in text.lower().replace("?", " ").split():
            word = word.strip(".,:").rstrip("s")
            if word and word not in STOP:
                row[hash_word(word) % DIM] += 1.0
        rows.append(row)
    return {"embeddings": rows}


def hash_word(word):
    """Stable across runs, unlike the built-in hash()."""
    value = 0
    for ch in word:
        value = (value * 31 + ord(ch)) % 1_000_003
    return value


class FakeChat:
    def __init__(self):
        self.calls = []

    def __call__(self, model, messages, options=None):
        self.calls.append(
            {"model": model, "messages": messages})
        reply = "It costs 6 dollars [1]."
        return {"message": {"content": reply}}


@pytest.fixture(scope="module")
def sample_pdf(tmp_path_factory):
    folder = tmp_path_factory.mktemp("docs")
    pdf = folder / "harbor_bikes.pdf"
    make_sample.write_pdf(make_sample.PAGES, pdf)
    return folder


@pytest.fixture(scope="module")
def index(sample_pdf):
    built = index_docs.build(sample_pdf, fake_embed)
    chunks, matrix, empty = built
    return chunks, matrix, empty


def test_pdf_pages_keep_numbers(sample_pdf):
    pages = pdf_text.load_folder(sample_pdf)
    assert [p["page"] for p in pages] == [1, 2, 3, 4]
    assert pages[0]["source"] == "harbor_bikes.pdf"
    assert "day pass costs 6 dollars" in pages[0]["text"]
    assert "Pine Avenue" in pages[1]["text"]
    assert pdf_text.empty_pages(pages) == []


def test_empty_pages_flags_scans():
    pages = [{"source": "s.pdf", "page": 1, "text": ""},
             {"source": "s.pdf", "page": 2, "text": "x " * 40}]
    assert pdf_text.empty_pages(pages)[0]["page"] == 1


def test_chunk_words_overlap():
    words = [f"w{i}" for i in range(30)]
    out = chunker.chunk_words(words, 12, 4)
    assert out[0].split()[-4:] == out[1].split()[:4]
    assert out[-1].split()[-1] == "w29"
    assert all(len(c.split()) <= 12 for c in out)


def test_chunk_words_short_and_empty():
    assert chunker.chunk_words(["a", "b"], 12, 4) == ["a b"]
    assert chunker.chunk_words([], 12, 4) == []


def test_chunk_words_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunker.chunk_words(["a"], 4, 4)


def test_chunks_keep_page_and_source(index):
    chunks, _, _ = index
    assert [c["id"] for c in chunks] == list(range(len(chunks)))
    assert {c["page"] for c in chunks} == {1, 2, 3, 4}
    assert len(chunks) > 4


def test_matrix_rows_have_length_one(index):
    chunks, matrix, _ = index
    assert matrix.shape == (len(chunks), DIM)
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)


def test_top_k_orders_by_cosine():
    matrix = emb.normalize(np.array([[1.0, 0.0], [1.0, 1.0],
                                     [0.0, 1.0]]))
    out = retrieve.top_k(np.array([2.0, 0.0]), matrix, k=2)
    assert [row for row, _ in out] == [0, 1]
    assert out[0][1] > out[1][1]
    assert out[1][1] == pytest.approx(0.7071, abs=1e-3)


def test_top_k_is_scale_free():
    matrix = emb.normalize(np.array([[3.0, 4.0], [4.0, 3.0]]))
    a = retrieve.top_k(np.array([1.0, 0.0]), matrix)
    b = retrieve.top_k(np.array([100.0, 0.0]), matrix)
    assert a[0][0] == b[0][0] == 1
    assert a[0][1] == pytest.approx(b[0][1])


def test_save_and_load_round_trip(index, tmp_path):
    chunks, matrix, _ = index
    emb.save_index(tmp_path / "idx", chunks, matrix)
    chunks2, matrix2 = emb.load_index(tmp_path / "idx")
    assert chunks2 == chunks
    assert np.array_equal(matrix2, matrix)


def test_load_refuses_other_embedding_model(index, tmp_path):
    chunks, matrix, _ = index
    folder = tmp_path / "idx"
    emb.save_index(folder, chunks, matrix)
    (folder / "meta.json").write_text(
        '{"model": "other-model", "count": 1}'
    )
    with pytest.raises(ValueError, match="rebuild"):
        emb.load_index(folder)


def test_embed_texts_batches(index):
    seen = []

    def spy(model, input):
        seen.append(len(input))
        return fake_embed(model, input)

    emb.embed_texts(["a b"] * 5, spy, batch=2)
    assert seen == [2, 2, 1]


def test_build_prompt_numbers_passages():
    hits = [{"text": "alpha", "source": "a.pdf", "page": 3},
            {"text": "beta", "source": "a.pdf", "page": 5}]
    prompt = ask.build_prompt("Why?", hits)
    assert "[1] alpha" in prompt and "[2] beta" in prompt
    assert prompt.endswith("Question: Why?")
    assert ask.format_sources(hits) == (
        "[1] a.pdf, page 3\n[2] a.pdf, page 5"
    )


def test_answer_sends_context_and_returns_hits(index):
    chunks, matrix, _ = index
    fake = FakeChat()
    reply, hits = ask.answer(
        "How much does a day pass cost?",
        chunks, matrix, fake, fake_embed,
    )
    assert reply == "It costs 6 dollars [1]."
    sent = fake.calls[0]["messages"]
    assert sent[0]["role"] == "system"
    assert "only" in sent[0]["content"]
    assert hits[0]["page"] == 1
    assert hits[0]["text"] in sent[1]["content"]


def test_ten_questions_find_their_pages(index):
    chunks, matrix, _ = index
    results = check.run_checks(chunks, matrix, fake_embed)
    missed = [q for q, _, ok in results if not ok]
    assert len(results) == 10
    assert missed == []


def test_unrelated_question_scores_low(index):
    chunks, matrix, _ = index
    on_topic = retrieve.search(
        "How much does a day pass cost?", chunks, matrix,
        fake_embed)
    off_topic = retrieve.search(
        "Does Harbor Street rent electric scooters?", chunks,
        matrix, fake_embed)
    assert off_topic[0]["score"] < on_topic[0]["score"]


def test_compare_sizes_reports_rank(sample_pdf):
    pages = pdf_text.load_folder(sample_pdf)
    count, rank = compare_sizes.rank_of_page(
        "When is the help line open?", 4, pages, fake_embed, 20
    )
    assert count > 4
    assert rank == 1


def test_summary_counts_chunks_per_page(index):
    chunks, matrix, _ = index
    info = inspect_index.summary(chunks, matrix)
    assert info["per_page"] == {
        ("harbor_bikes.pdf", 1): 3,
        ("harbor_bikes.pdf", 2): 2,
        ("harbor_bikes.pdf", 3): 2,
        ("harbor_bikes.pdf", 4): 2,
    }
    assert info["shape"] == (9, DIM)
    assert info["shortest"] == pytest.approx(1.0, abs=1e-5)
    assert info["longest"] == pytest.approx(1.0, abs=1e-5)


def test_similarity_of_a_row_with_itself_is_one(index):
    _, matrix, _ = index
    value = inspect_index.similarity(matrix, 0, 0)
    assert value == pytest.approx(1.0, abs=1e-5)
