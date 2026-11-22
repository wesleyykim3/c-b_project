"""Tests for common/score.py. Run from the repo root:

    /opt/miniconda3/bin/python -m unittest discover -s tests -t .
"""

import unittest

from common.score import score_row

OBJS = ["guitar", "kettle", "ladder", "violin", "camera"]
NAMES = ["Paul", "Ann", "Mark", "Lucy", "John"]
FACTS = [{"position": i + 1, "name": n, "obj": o}
         for i, (n, o) in enumerate(zip(NAMES, OBJS))]


def trial(output, **extra):
    return {"model": "m", "list_id": 0, "list_length": 5, "condition": "none",
            "facts": FACTS, "raw_output": output, **extra}


def lines(*pairs):
    return "\n".join(f"{n}: {o}" for n, o in pairs)


def free(rows):
    return [r["free_correct"] for r in rows]


def serial(rows):
    return [r["serial_correct"] for r in rows]


class ScoreTests(unittest.TestCase):
    def test_perfect_order_both_scorers_full(self):
        rows, s = score_row(trial(lines(*zip(NAMES, OBJS))))
        self.assertEqual(free(rows), [True] * 5)
        self.assertEqual(serial(rows), [True] * 5)
        self.assertEqual([r["recalled_rank"] for r in rows], [1, 2, 3, 4, 5])
        self.assertEqual((s["free_acc"], s["serial_acc"]), (1.0, 1.0))

    def test_reversed_order_free_full_serial_only_middle(self):
        pairs = list(zip(NAMES, OBJS))[::-1]
        rows, s = score_row(trial(lines(*pairs)))
        self.assertEqual(free(rows), [True] * 5)
        self.assertEqual(serial(rows), [False, False, True, False, False])

    def test_omission_shifts_later_ranks(self):
        # Omit position 2: later items are all recalled but one rank early.
        pairs = [p for i, p in enumerate(zip(NAMES, OBJS)) if i != 1]
        rows, s = score_row(trial(lines(*pairs)))
        self.assertEqual(free(rows), [True, False, True, True, True])
        self.assertEqual(serial(rows), [True, False, False, False, False])
        self.assertEqual(s["omissions"], 1)

    def test_intrusion_occupies_a_rank(self):
        pairs = [("Paul", "guitar"), ("Zelda", "kettle"), ("Ann", "kettle")]
        rows, s = score_row(trial(lines(*pairs)))
        self.assertEqual(s["intrusions"], 1)
        # Ann is at rank 3, not 2: free yes, serial no.
        self.assertTrue(rows[1]["free_correct"])
        self.assertFalse(rows[1]["serial_correct"])
        self.assertEqual(rows[1]["recalled_rank"], 3)

    def test_repeats_do_not_shift_ranks(self):
        pairs = [("Paul", "guitar"), ("Paul", "guitar"), ("Paul", "guitar"),
                 ("Ann", "kettle")]
        rows, s = score_row(trial(lines(*pairs)))
        self.assertTrue(rows[1]["serial_correct"])     # Ann still rank 2
        self.assertEqual(s["repeats"], 2)

    def test_first_occurrence_wrong_object_not_rescued_by_later_repeat(self):
        pairs = [("Paul", "kettle"), ("Paul", "guitar")]
        rows, s = score_row(trial(lines(*pairs)))
        self.assertFalse(rows[0]["free_correct"])
        self.assertFalse(rows[0]["serial_correct"])
        self.assertEqual(s["wrong_object"], 1)

    def test_wrong_object_counts_as_recalled_not_omitted(self):
        rows, s = score_row(trial(lines(("Paul", "kettle"))))
        self.assertEqual(rows[0]["recalled_rank"], 1)
        self.assertEqual(s["omissions"], 4)
        self.assertEqual(s["wrong_object"], 1)

    def test_empty_output(self):
        rows, s = score_row(trial(""))
        self.assertEqual(free(rows), [False] * 5)
        self.assertEqual(serial(rows), [False] * 5)
        self.assertTrue(all(r["recalled_rank"] is None for r in rows))
        self.assertEqual((s["omissions"], s["parse_rate"]), (5, 0.0))

    def test_metadata_passes_through_and_payload_does_not(self):
        rows, s = score_row(trial("", filler_items=6, prompt_variant="serial",
                                  cached=True))
        for d in (rows[0], s):
            self.assertEqual(d["model"], "m")
            self.assertEqual(d["filler_items"], 6)
            self.assertEqual(d["prompt_variant"], "serial")
            for k in ("facts", "raw_output", "cached"):
                self.assertNotIn(k, d)
        self.assertEqual([r["position"] for r in rows], [1, 2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
