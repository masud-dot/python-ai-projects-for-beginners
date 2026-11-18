"""Write reviews.csv: 40 short reviews I wrote for this book.

human_label is my own reading of each review. Row 39 is empty
on purpose, and row 40 is long, so the cleaning and truncation
steps have something to do. Runs offline; needs only pandas.
"""
from pathlib import Path

import pandas as pd

P = "POSITIVE"
N = "NEGATIVE"
ROWS = [
    ("The battery lasts two full days and the screen is bright.", P),
    ("Arrived broken, and support never answered my emails.", N),
    ("Best coffee grinder I have owned. Quiet and even.", P),
    ("The zipper failed after one week. Very disappointed.", N),
    ("Fast shipping and the fit is perfect.", P),
    ("It works, but the setup took me three hours.", N),
    ("I love how light this laptop is.", P),
    ("The sound is thin and tinny. I returned it.", N),
    ("Great value for the price, and the colors are lovely.", P),
    ("Stopped charging after a month. Do not buy.", N),
    ("The waiter was kind and the pasta was excellent.", P),
    ("Cold food, slow service, and a rude manager.", N),
    ("A beautiful, funny film. I smiled the whole way through.", P),
    ("The plot made no sense and the acting was flat.", N),
    ("My kids use it every day and it still looks new.", P),
    ("The instructions are confusing and a part was missing.", N),
    ("Comfortable chair; my back pain is much better.", P),
    ("Smells like chemicals and the paint is already peeling.", N),
    ("Five stars. I would order from this shop again.", P),
    ("Not worth the money. It feels cheap and flimsy.", N),
    ("The app is simple and does exactly what I need.", P),
    ("Crashes every time I open it. Useless.", N),
    ("Wonderful hotel, spotless room, friendly staff.", P),
    ("The room was dirty and the walls were paper thin.", N),
    ("This book kept me up all night. Brilliant ending.", P),
    ("I gave up on page 50. Boring and far too long.", N),
    ("Not bad at all. I expected worse for this price.", P),
    ("I can't say I'm unhappy with it.", P),
    ("Oh great, another update that deletes my settings.", N),
    ("Wow, thanks for the two-hour wait. Truly a treat.", N),
    ("Fantastic. It broke on day one, just like my last one.", N),
    ("Not good, not terrible. It is a phone.", N),
    ("The camera is superb, but the battery barely lasts a day.", N),
    ("Terrible smell at first, but it faded and I love it now.", P),
    ("Delivery was late, though the product itself is great.", P),
    ("I wanted to love it. I really did.", N),
    ("The staff at the Italian place were warm and quick.", P),
    ("Nothing special, but it does the job fine.", P),
    ("", N),
]
LONG = (
    "The box was big and the packaging was careful, with foam "
    "around every part and a printed card on top. Everything "
    "arrived on the date promised and the courier was polite. "
)
ENDING = "Sadly the product itself broke on the first day."


def main():
    rows = [(i, t, h) for i, (t, h) in enumerate(ROWS, 1)]
    long_text = LONG * 40 + ENDING
    rows.append((len(rows) + 1, long_text, N))
    df = pd.DataFrame(rows, columns=["id", "text", "human_label"])
    df.loc[df["text"] == "", "text"] = None
    df.to_csv(Path(__file__).parent / "reviews.csv", index=False)
    print(f"wrote {len(df)} rows")


if __name__ == "__main__":
    main()
