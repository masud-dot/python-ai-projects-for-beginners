from types import SimpleNamespace as NS

import language_report as lr
import sentences as s


def test_sentences_and_language_report():
    segs = [NS(text=" Hello there. How are"), NS(text=" you? Fine.")]
    assert s.sentences(segs) == ["Hello there.", "How are you?", "Fine."]
    low = NS(language="de", language_probability=0.55)
    high = NS(language="en", language_probability=0.99)
    assert "WARNING" in lr.report("a.wav", low)
    assert "WARNING" not in lr.report("b.wav", high)
