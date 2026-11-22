"""Tests for common/parse.py. Run from the repo root:

    /opt/miniconda3/bin/python -m unittest discover -s tests -t .
"""

import unittest

from common.generate import Fact
from common.parse import parse_recall

FACTS = [
    Fact(1, "Paul", "has-a", "guitar", "Paul has a guitar."),
    Fact(2, "Ann", "has-a", "kettle", "Ann has a kettle."),
    Fact(3, "Mark", "has-a", "guitar", "Mark has a guitar."),
]


class ParseTests(unittest.TestCase):
    def test_clean_output_keeps_order_and_positions(self):
        r = parse_recall("Mark: guitar\nPaul: guitar\nAnn: kettle", FACTS)
        self.assertEqual([e.matched_position for e in r.events], [3, 1, 2])
        self.assertEqual([e.rank for e in r.events], [1, 2, 3])
        self.assertTrue(all(e.correct for e in r.events))
        self.assertEqual(r.parse_rate, 1.0)

    def test_wrong_object_is_matched_but_incorrect(self):
        r = parse_recall("Paul: kettle", FACTS)
        e = r.events[0]
        self.assertEqual(e.matched_position, 1)
        self.assertFalse(e.correct)

    def test_intrusion_has_no_position(self):
        r = parse_recall("Zelda: guitar", FACTS)
        e = r.events[0]
        self.assertIsNone(e.matched_position)
        self.assertFalse(e.correct)

    def test_repeat_flagged_and_kept(self):
        r = parse_recall("Paul: guitar\nAnn: kettle\nPaul: guitar", FACTS)
        self.assertEqual([e.repeat for e in r.events], [False, False, True])
        self.assertEqual(len(r.events), 3)

    def test_bullets_numbering_markdown_case_dashes(self):
        text = "- Paul: guitar\n2. **ann**: Kettle.\n3) MARK - guitar,\n* `Paul`: guitar"
        r = parse_recall(text, FACTS)
        self.assertEqual([e.name for e in r.events], ["paul", "ann", "mark", "paul"])
        self.assertTrue(all(e.correct for e in r.events))
        self.assertEqual(r.parse_rate, 1.0)

    def test_prose_and_arithmetic_lower_parse_rate_without_intrusions(self):
        text = ("Sure, here are the facts:\n"
                "Paul: guitar\n"
                "\n"
                "What is 12 + 5? Answer: 17\n"
                "Ann: kettle (position 2)\n"
                "Note: I hope this helps")
        r = parse_recall(text, FACTS)
        # "Note: I hope ..." and the others must not parse as intrusions.
        self.assertEqual([e.name for e in r.events], ["paul"])
        self.assertEqual(r.n_lines, 5)
        self.assertAlmostEqual(r.parse_rate, 1 / 5)

    def test_empty_and_none(self):
        for text in ("", "   \n\n", None):
            r = parse_recall(text, FACTS)
            self.assertEqual(r.events, [])
            self.assertEqual(r.parse_rate, 0.0)

    def test_accepts_dict_facts_from_raw_jsonl(self):
        dicts = [{"position": f.position, "name": f.name, "obj": f.obj} for f in FACTS]
        r = parse_recall("Ann: kettle", dicts)
        self.assertEqual(r.events[0].matched_position, 2)
        self.assertTrue(r.events[0].correct)


if __name__ == "__main__":
    unittest.main()
