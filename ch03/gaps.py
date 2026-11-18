import numpy as np


def inject_gaps(df, cols, frac=0.02, seed=0):
    """Return a copy with about frac of each column blanked."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    for col in cols:
        hit = rng.random(len(out)) < frac
        out.loc[hit, col] = np.nan
    return out
