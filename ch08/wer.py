"""Word error rate between a reference and a hypothesis."""
import re
import sys


def words(text):
    text = text.lower().replace("'", "")
    return re.sub(r"[^a-z0-9]+", " ", text).split()


def edit_counts(ref, hyp):
    """Return (substitutions, deletions, insertions)."""
    # cost[i][j] = (total, subs, dels, ins) for ref[:i], hyp[:j]
    prev = [(j, 0, 0, j) for j in range(len(hyp) + 1)]
    for i in range(1, len(ref) + 1):
        row = [(i, 0, i, 0)]
        for j in range(1, len(hyp) + 1):
            t, s, d, n = prev[j - 1]
            if ref[i - 1] != hyp[j - 1]:
                t, s = t + 1, s + 1
            best = (t, s, d, n)
            t, s, d, n = prev[j]
            if t + 1 < best[0]:
                best = (t + 1, s, d + 1, n)
            t, s, d, n = row[j - 1]
            if t + 1 < best[0]:
                best = (t + 1, s, d, n + 1)
            row.append(best)
        prev = row
    return prev[-1][1:]


def wer(reference, hypothesis):
    ref, hyp = words(reference), words(hypothesis)
    subs, dels, ins = edit_counts(ref, hyp)
    return (subs + dels + ins) / len(ref), (subs, dels, ins)


if __name__ == "__main__":
    ref = open(sys.argv[1], encoding="utf-8").read()
    hyp = open(sys.argv[2], encoding="utf-8").read()
    rate, (s, d, i) = wer(ref, hyp)
    print(f"Reference words: {len(words(ref))}")
    print(f"Substitutions {s}, deletions {d}, insertions {i}")
    print(f"WER: {rate:.1%}")
