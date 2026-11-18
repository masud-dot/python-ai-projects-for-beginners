import random


def summarize(seed=0, n=20):
    random.seed(seed)
    scores = [random.randint(40, 100) for _ in range(n)]
    average = sum(scores) / len(scores)
    above = [s for s in scores if s > average]
    return average, max(scores), len(above)


if __name__ == "__main__":
    average, best, count = summarize()
    print(f"Average score: {average:.1f}")
    print(f"Best score: {best}")
    print(f"{count} of 20 scores beat the average")
