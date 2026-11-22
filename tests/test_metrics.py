"""Tests for common/metrics.py. Run from the repo root:

    /opt/miniconda3/bin/python -m unittest discover -s tests -t .
"""

import unittest
import numpy as np

from common.score import score_rows
from common.metrics import (group_trials, spc, pfr, no_recall_rate, rpr,
                            bootstrap_ci)

NAMES = ["Paul", "Ann", "Mark", "Lucy", "John"]
OBJS = ["guitar", "kettle", "ladder", "violin", "camera"]
FACTS = [{"position": i + 1, "name": n, "obj": o}
         for i, (n, o) in enumerate(zip(NAMES, OBJS))]


def raw(order, **extra):
    """Raw trial whose output recalls the given 1-indexed positions in order.
    Use a (name, obj) tuple instead of an int for arbitrary entries."""
    out = []
    for x in order:
        out.append(f"{x[0]}: {x[1]}" if isinstance(x, tuple)
                   else f"{NAMES[x - 1]}: {OBJS[x - 1]}")
    return {"model": "m", "list_id": 0, "list_length": 5, "condition": "none",
            "facts": FACTS, "raw_output": "\n".join(out), **extra}


def trials_for(*orders):
    rows, _ = score_rows([raw(o) for o in orders])
    (cell,) = group_trials(rows, ("model",)).values()
    return cell


class MetricTests(unittest.TestCase):
    def test_group_trials_uses_trial_id_and_cell_keys(self):
        rows, _ = score_rows([raw([1], list_id=0), raw([2], list_id=0),
                              raw([3], condition="after_list")])
        cells = group_trials(rows, ("condition",))
        self.assertEqual({k: len(v) for k, v in cells.items()},
                         {("none",): 2, ("after_list",): 1})
        for trial in cells[("none",)]:
            self.assertEqual([r["position"] for r in trial], [1, 2, 3, 4, 5])

    def test_spc_free_and_serial(self):
        ts = trials_for([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
        np.testing.assert_allclose(spc(ts, "free"), [1, 1, 1, 1, 1])
        np.testing.assert_allclose(spc(ts, "serial"), [.5, .5, 1, .5, .5])
        with self.assertRaises(ValueError):
            spc(ts, "bogus")

    def test_pfr_counts_first_correct_recall(self):
        ts = trials_for([1, 2], [1, 3], [5, 1])
        np.testing.assert_allclose(pfr(ts), [2 / 3, 0, 0, 0, 1 / 3])

    def test_pfr_skips_intrusion_and_drops_empty_trials(self):
        ts = trials_for([("Zelda", "x"), 3], [("Zelda", "x")], [2])
        # Trial 1's first RECALLED fact is 3; trial 2 recalled nothing.
        np.testing.assert_allclose(pfr(ts), [0, .5, .5, 0, 0])
        np.testing.assert_allclose(pfr(trials_for([("Zelda", "x")])), [np.nan] * 5)

    def test_no_recall_rate_counts_the_trials_pfr_drops(self):
        # empty output, intrusion only, wrong object only, and two real recalls
        ts = trials_for([], [("Zelda", "x")], [("Paul", "kettle")], [1], [3])
        self.assertAlmostEqual(no_recall_rate(ts), 3 / 5)
        np.testing.assert_allclose(pfr(ts), [.5, 0, .5, 0, 0])   # only 2 trials counted
        self.assertEqual(no_recall_rate(trials_for([1], [2])), 0.0)
        lo, hi = bootstrap_ci(ts, no_recall_rate, n_boot=50)
        self.assertTrue(0 <= lo <= hi <= 1)

    def test_rpr(self):
        ts = trials_for([1, 2, 3, 4, 5], [5])      # spc = [.5 .5 .5 .5 1]
        self.assertAlmostEqual(rpr(ts), 2.0)       # default k = 1 for L = 5
        self.assertAlmostEqual(rpr(ts, k=2), 0.75 / 0.5)
        self.assertTrue(np.isnan(rpr(trials_for([5]))))   # primacy accuracy 0
        with self.assertRaises(ValueError):
            rpr(ts, k=3)                            # ends would overlap

    def test_mixed_lengths_rejected(self):
        short = [{"free_correct": True, "serial_correct": True, "position": 1,
                  "recalled_rank": 1}]
        with self.assertRaises(ValueError):
            spc([trials_for([1])[0], short])

    def test_bootstrap_ci_collapses_for_identical_trials(self):
        ts = trials_for([1, 2, 3, 4, 5]) * 4
        lo, hi = bootstrap_ci(ts, spc, n_boot=50)
        np.testing.assert_allclose(lo, 1.0)
        np.testing.assert_allclose(hi, 1.0)

    def test_bootstrap_ci_brackets_point_estimate_and_is_seeded(self):
        ts = trials_for(*([[1, 2, 3, 4, 5]] * 10 + [[5]] * 10))
        lo, hi = bootstrap_ci(ts, spc, n_boot=200, seed=1)
        pt = spc(ts)
        self.assertTrue(np.all(lo <= pt + 1e-9) and np.all(pt <= hi + 1e-9))
        self.assertTrue(np.any(hi > lo))
        lo2, hi2 = bootstrap_ci(ts, spc, n_boot=200, seed=1)
        np.testing.assert_array_equal(lo, lo2)

    def test_bootstrap_ci_for_array_and_scalar_metrics(self):
        ts = trials_for(*([[1, 2, 3]] * 5 + [[3, 2, 1]] * 5))
        lo, hi = bootstrap_ci(ts, pfr, n_boot=50)
        self.assertEqual(lo.shape, (5,))           # one value per position
        lo, hi = bootstrap_ci(ts, rpr, n_boot=50)
        self.assertEqual(np.ndim(lo), 0)


if __name__ == "__main__":
    unittest.main()
