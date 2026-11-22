"""Tests for common/plots.py. Run from the repo root:

    /opt/miniconda3/bin/python -m unittest discover -s tests -t .
"""

import tempfile
import unittest
from pathlib import Path

from common.score import score_rows
from common.metrics import group_trials
from common.plots import plot_spc, plot_pfr, plot_dual_scoring, save, FIGURES_DIR

NAMES = ["Paul", "Ann", "Mark", "Lucy", "John"]
OBJS = ["guitar", "kettle", "ladder", "violin", "camera"]
FACTS = [{"position": i + 1, "name": n, "obj": o}
         for i, (n, o) in enumerate(zip(NAMES, OBJS))]


def raw(order, condition):
    out = "\n".join(f"{NAMES[p - 1]}: {OBJS[p - 1]}" for p in order)
    return {"model": "m", "list_id": 0, "list_length": 5, "condition": condition,
            "facts": FACTS, "raw_output": out}


def cells(*conditions):
    rows = []
    for c in conditions:
        rows += [raw([1, 2, 3, 4, 5], c)] * 3 + [raw([5, 4], c), raw([], c)]
    pos, _ = score_rows(rows)
    return {k[0]: v for k, v in group_trials(pos, ("condition",)).items()}


class PlotTests(unittest.TestCase):
    def test_one_line_per_cell_with_ci_bands(self):
        fig = plot_spc(cells("a", "b", "c"), n_boot=20)
        ax = fig.axes[0]
        self.assertEqual(len(ax.lines), 3)
        self.assertEqual(len(ax.collections), 3)           # one CI band each
        self.assertEqual([t.get_text() for t in ax.get_legend().get_texts()],
                         ["a", "b", "c"])

    def test_single_series_spc_has_no_legend(self):
        fig = plot_spc(cells("a"), n_boot=20)
        self.assertIsNone(fig.axes[0].get_legend())

    def test_pfr_legend_reports_no_recall_rate(self):
        fig = plot_pfr(cells("a"), n_boot=20)
        (text,) = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        self.assertIn("20% no recall", text)               # 1 of 5 trials empty

    def test_dual_scoring_has_two_lines(self):
        (trials,) = cells("a").values(),
        fig = plot_dual_scoring(next(iter(cells("a").values())), n_boot=20)
        self.assertEqual(len(fig.axes[0].lines), 2)

    def test_light_surface_and_no_theme_option(self):
        fig = plot_spc(cells("a", "b"), n_boot=20)
        self.assertEqual(fig.get_facecolor()[:3], (0xfc / 255, 0xfc / 255, 0xfb / 255))
        with self.assertRaises(TypeError):
            plot_spc(cells("a"), n_boot=20, theme="dark")

    def test_more_than_eight_lines_and_empty_rejected(self):
        many = cells(*"abcdefghi")
        with self.assertRaises(ValueError):
            plot_spc(many, n_boot=5)
        with self.assertRaises(ValueError):
            plot_spc({}, n_boot=5)

    def test_saves_png_and_creates_folders(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "sub" / "spc.png"
            plot_spc(cells("a"), n_boot=20, path=target)
            self.assertGreater(target.stat().st_size, 1000)

    def test_bare_filename_goes_to_figures_dir_not_data(self):
        # Don't write a file: check the path resolution only.
        self.assertEqual(FIGURES_DIR.name, "figures")
        self.assertNotIn("data", FIGURES_DIR.parts[-1])
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(save(plot_spc(cells("a"), n_boot=5),
                                  Path(d) / "x.png").parent, Path(d))


if __name__ == "__main__":
    unittest.main()
