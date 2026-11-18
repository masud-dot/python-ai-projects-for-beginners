import random

random.seed(0)
scores = [random.randint(40, 100) for _ in range(20)]
average = sum(scores) / len(scores)
best = max(scores)
above = [s for s in scores if s > average]

print(f"Average score: {average:.1f}")
print(f"Best score: {best}")
print(f"{len(above)} of {len(scores)} scores beat the average")
