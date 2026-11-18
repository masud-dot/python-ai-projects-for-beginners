# Chapter 10 tests. They run offline: the Ollama server is
# replaced by a fake chat function that keeps no state, as the
# real server does. first_chat.py, the input loops and the
# real model need Ollama, so they are not run here.
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import chat as chat_mod  # noqa: E402
import check  # noqa: E402
import fails  # noqa: E402
import first_chat  # noqa: E402
import history as hist  # noqa: E402
import memory_chat  # noqa: E402
import stream_chat  # noqa: E402


class FakeChat:
    """Stateless fake: it only knows what is in `messages`."""

    def __init__(self):
        self.calls = []

    def reply_for(self, messages):
        text = " ".join(m["content"] for m in messages)
        last = messages[-1]["content"].lower()
        if "cat called" in last or "my name" in last:
            if "Biscuit" in text:
                return "Your cat is called Biscuit."
            if "Priya" in text:
                return "Your name is Priya."
            return "I do not know."
        if messages[0]["content"].startswith("Always end"):
            return "Red. -- Ollie"
        return "Fine."

    def __call__(self, model, messages, stream=False, options=None):
        self.calls.append(
            {"model": model, "messages": [dict(m) for m in messages],
             "stream": stream, "options": options}
        )
        text = self.reply_for(messages)
        if not stream:
            return {"message": {"role": "assistant", "content": text}}
        return self.chunks(text)

    @staticmethod
    def chunks(text):
        words = text.split(" ")
        for i, w in enumerate(words):
            piece = w + ("" if i == len(words) - 1 else " ")
            yield {"message": {"content": piece}, "done": False}
        yield {
            "message": {"content": ""},
            "done": True,
            "eval_count": 40,
            "eval_duration": 2_000_000_000,
        }


def test_ask_sends_one_user_message():
    fake = FakeChat()
    out = first_chat.ask("Hi", fake)
    assert out == "Fine."
    assert fake.calls[0]["messages"] == [
        {"role": "user", "content": "Hi"}
    ]


def test_send_resends_whole_history():
    fake = FakeChat()
    h = memory_chat.new_history()
    memory_chat.send(h, "My cat is called Biscuit.", fake)
    memory_chat.send(h, "What is my cat called?", fake)
    second = fake.calls[1]["messages"]
    assert [m["role"] for m in second] == [
        "system", "user", "assistant", "user"
    ]
    assert h[-1]["content"] == "Your cat is called Biscuit."


def test_tokens_per_second():
    chunk = {"eval_count": 40, "eval_duration": 2_000_000_000}
    assert stream_chat.tokens_per_second(chunk) == 20.0


def test_stream_reply_joins_pieces(capsys):
    fake = FakeChat()
    h = memory_chat.new_history()
    h.append({"role": "user", "content": "hello"})
    text, speed = stream_chat.stream_reply(h, fake)
    assert text == "Fine."
    assert speed == 20.0
    assert capsys.readouterr().out == "Fine.\n"
    assert fake.calls[0]["stream"] is True


def pairs(n):
    msgs = [{"role": "system", "content": "SYS"}]
    for i in range(n):
        msgs.append({"role": "user", "content": f"q{i} " * 40})
        msgs.append({"role": "assistant", "content": f"a{i} " * 40})
    msgs.append({"role": "user", "content": "latest"})
    return msgs


def test_trim_noop_when_it_fits():
    msgs = pairs(2)
    assert hist.trim_history(msgs, budget=10_000) == msgs


def test_trim_drops_oldest_pairs_keeps_system_and_newest():
    msgs = pairs(6)
    out = hist.trim_history(msgs, budget=300)
    assert out[0] == msgs[0]
    assert out[-1] == msgs[-1]
    assert len(out) < len(msgs)
    assert out[1]["role"] == "user"
    assert out[1] in msgs[3:]
    assert hist.history_tokens(out) <= 300
    assert len(msgs) == 14  # the input list is not changed


def test_trim_never_empties_the_chat():
    msgs = pairs(3)
    out = hist.trim_history(msgs, budget=1)
    assert out == [msgs[0], msgs[-1]]


def test_trim_without_system_prompt():
    msgs = pairs(3)[1:]
    out = hist.trim_history(msgs, budget=100)
    assert out[-1]["content"] == "latest"
    assert out[0]["role"] == "user"


def test_run_turn_trims_before_sending():
    fake = FakeChat()
    h = memory_chat.new_history()
    for i in range(8):
        chat_mod.run_turn(h, f"fact {i} " * 30, fake, 200, False)
    sent = fake.calls[-1]["messages"]
    assert hist.history_tokens(sent) <= 200
    assert sent[0]["role"] == "system"


def test_checks_pass_with_memory_and_fail_without():
    assert check.check_memory(FakeChat())
    assert check.check_system_prompt(FakeChat())
    assert check.check_no_memory(FakeChat())

    def forgetful(model, messages, stream=False, options=None):
        return {"message": {"content": "I do not know."}}

    assert not check.check_memory(forgetful)


def test_long_chat_forgets_the_first_turn():
    first, reply = fails.long_chat(FakeChat(), budget=40)
    assert "Priya" not in first
    assert "do not know" in reply


def test_speed_measure():
    import speed

    assert speed.measure(FakeChat()) == 20.0
