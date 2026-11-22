"""Serial-position metrics.

Works on the per-position rows from common.score.score_rows. A "trial" here is
the list of its position rows sorted by position (use group_trials to build
them). All metrics take a list of trials that share one list length, i.e. one
experimental cell, and return numpy arrays indexed by position (or a single float).

  spc          accuracy by serial position, under free or serial scoring
  pfr          probability of first recall by position
  no_recall_rate  share of trials with no correct recall (the trials pfr
                  drops; always report it next to pfr)
  rpr          recency-to-primacy ratio
  bootstrap_ci percentile CI for any of the above, resampling lists (trials)
"""

from __future__ import annotations
import warnings
from collections import defaultdict
import numpy as np


def group_trials(pos_rows, keys) -> dict:
    """Group position rows into trials, then trials into cells.

    keys: row fields that define a cell, e.g. ("model", "condition",
    "list_length"). Returns {cell_key_tuple: [trial, ...]} where each trial is
    its position rows sorted by position.
    """
    by_trial = defaultdict(list)
    for r in pos_rows:
        by_trial[r["trial_id"]].append(r)
    cells = defaultdict(list)
    for rows in by_trial.values():
        rows.sort(key=lambda r: r["position"])
        cells[tuple(rows[0][k] for k in keys)].append(rows)
    return dict(cells)


def _length(trials) -> int:
    lengths = {len(t) for t in trials}
    if len(lengths) != 1:
        raise ValueError(f"trials must share one list length, got {sorted(lengths)}")
    return lengths.pop()


def spc(trials, scoring: str = "free") -> np.ndarray:
    """Accuracy at each serial position (index 0 = position 1)."""
    if scoring not in ("free", "serial"):
        raise ValueError(f"scoring must be 'free' or 'serial', got {scoring!r}")
    field = f"{scoring}_correct"
    return np.array([[r[field] for r in t] for t in trials], dtype=float).mean(axis=0)


def _recall_sequence(trial) -> list[int]:
    """Positions of correctly recalled facts, in output order."""
    got = [(r["recalled_rank"], r["position"]) for r in trial if r["free_correct"]]
    return [pos for _, pos in sorted(got)]


def pfr(trials) -> np.ndarray:
    """P(position p is the first correctly recalled fact).

    Trials with no correct recall are dropped from the denominator, so the
    result sums to 1 (or is all-nan if no trial recalled anything). Always
    report no_recall_rate(trials) alongside it: a cell where many trials are
    dropped has a PFR that describes only the trials that recalled something.
    """
    L = _length(trials)
    counts = np.zeros(L)
    n = 0
    for t in trials:
        seq = _recall_sequence(t)
        if seq:
            counts[seq[0] - 1] += 1
            n += 1
    return counts / n if n else np.full(L, np.nan)


def no_recall_rate(trials) -> float:
    """Proportion of trials with NO correctly recalled fact: the trials that
    pfr drops. Works with bootstrap_ci."""
    return float(np.mean([not _recall_sequence(t) for t in trials]))


def rpr(trials, k: int | None = None, scoring: str = "free") -> float:
    """Recency-to-primacy ratio: mean accuracy of the last k positions divided
    by that of the first k. k defaults to round(20% of list length), min 1.
    nan if primacy accuracy is 0."""
    curve = spc(trials, scoring)
    L = len(curve)
    k = k or max(1, round(0.2 * L))
    if not 1 <= k <= L // 2:
        raise ValueError(f"k={k} must be between 1 and L//2={L // 2} so the ends don't overlap")
    pri = curve[:k].mean()
    return float(curve[-k:].mean() / pri) if pri > 0 else float("nan")


def bootstrap_ci(trials, metric, n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 0, **kwargs):
    """Percentile bootstrap CI for metric(trials, **kwargs), resampling LISTS
    (trials) with replacement. Returns (lo, hi) with the same shape as the
    metric"""
    rng = np.random.default_rng(seed)
    n = len(trials)

    boots = np.stack([
        np.asarray(metric([trials[i] for i in rng.integers(0, n, n)], **kwargs),
                   dtype=float)
        for _ in range(n_boot)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-nan slice -> nan, as intended
        lo = np.nanpercentile(boots, 100 * alpha / 2, axis=0)
        hi = np.nanpercentile(boots, 100 * (1 - alpha / 2), axis=0)
    return lo, hi
