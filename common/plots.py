"""Plots for the serial-position metrics.

Every function takes `cells`: {label: [trial, ...]}, one entry per line on the
chart (a condition, a model, a prompt variant...). Build trials/cells with
common.metrics.group_trials. Each line gets a bootstrap CI band over lists.

  plot_spc          accuracy by serial position (free or serial scoring)
  plot_pfr          probability of first recall; legend shows each cell's
                    no-recall rate, because pfr drops those trials
  plot_dual_scoring free vs strict-serial accuracy for ONE cell

Each returns a matplotlib Figure and default output folder is figures/.
Light theme only.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
from matplotlib.figure import Figure

from common.metrics import spc, pfr, no_recall_rate, bootstrap_ci

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

# Categorical slots 1-8, fixed order (validated light-surface palette).
_SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#6250d6", "#e34948"]
_SURFACE, _TEXT, _MUTED, _GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def _new_axes(title: str | None, xlabel: str, ylabel: str):
    fig = Figure(figsize=(7.5, 4.5), dpi=150, facecolor=_SURFACE)
    ax = fig.subplots()
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.grid(axis="y", color=_GRID, linewidth=0.8)      # hairline, solid, recessive
    ax.set_axisbelow(True)
    ax.tick_params(colors=_MUTED, labelsize=9, length=3)
    ax.set_xlabel(xlabel, color=_MUTED, fontsize=10)
    ax.set_ylabel(ylabel, color=_MUTED, fontsize=10)
    if title:
        ax.set_title(title, color=_TEXT, fontsize=12, loc="left", pad=12)
    return fig, ax


def _legend(ax, ncol):
    leg = ax.legend(frameon=False, fontsize=9, labelcolor=_TEXT,
                    loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=ncol)
    for h in leg.get_lines():
        h.set_linewidth(2.2)


def _draw(cells, metric, label_fn, title, xlabel, ylabel, ylim, n_boot, path,
          **metric_kwargs):
    if not cells:
        raise ValueError("cells is empty")
    if len(cells) > len(_SERIES):
        raise ValueError(f"{len(cells)} lines; the palette has {len(_SERIES)} slots. "
                         "Fold into 'Other' or facet instead.")
    fig, ax = _new_axes(title, xlabel, ylabel)
    for color, (name, trials) in zip(_SERIES, cells.items()):
        y = metric(trials, **metric_kwargs)
        lo, hi = bootstrap_ci(trials, metric, n_boot=n_boot, **metric_kwargs)
        x = np.arange(1, len(y) + 1)
        ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.plot(x, y, color=color, linewidth=1.6, solid_capstyle="round",
                solid_joinstyle="round", label=label_fn(name, trials))
    ax.set_ylim(*ylim)
    ax.margins(x=0.02)
    if len(cells) >= 2 or label_fn is not _plain_label:
        _legend(ax, ncol=min(len(cells), 3))
    fig.tight_layout()
    if path is not None:
        save(fig, path)
    return fig


def _plain_label(name, trials):
    return str(name)


def _pfr_label(name, trials):
    return f"{name}  ({no_recall_rate(trials):.0%} no recall)"


def plot_spc(cells, scoring: str = "free", title: str | None = None,
             n_boot: int = 1000, path=None) -> Figure:
    """Serial position curve, one line per cell, with 95% bootstrap bands."""
    return _draw(cells, spc, _plain_label,
                 title or f"Serial position curve ({scoring} scoring)",
                 "Serial position", "Accuracy", (0, 1.02), n_boot, path,
                 scoring=scoring)


def plot_pfr(cells, title: str | None = None, n_boot: int = 1000,
             path=None) -> Figure:
    """Probability of first recall. The legend carries each cell's no-recall
    rate (trials dropped from the PFR), so it is always reported with the curve."""
    ymax = max(0.05, max(float(np.nanmax(pfr(ts))) for ts in cells.values())) \
        if cells else 1.0
    return _draw(cells, pfr, _pfr_label, title or "Probability of first recall",
                 "Serial position", "P(first recall)", (0, min(1.02, ymax * 1.15)),
                 n_boot, path)


def plot_dual_scoring(trials, title: str | None = None, n_boot: int = 1000,
                      path=None) -> Figure:
    """Free-recall vs strict-serial accuracy for one cell. The gap between the
    lines is the dual-scoring dissociation."""
    if not trials:
        raise ValueError("trials is empty")
    fig, ax = _new_axes(title or "Free recall vs strict serial scoring",
                        "Serial position", "Accuracy")
    names = {"free": "Free recall", "serial": "Strict serial"}
    for color, scoring in zip(_SERIES, names):
        y = spc(trials, scoring)
        lo, hi = bootstrap_ci(trials, spc, n_boot=n_boot, scoring=scoring)
        x = np.arange(1, len(y) + 1)
        ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.plot(x, y, color=color, linewidth=1.6, solid_capstyle="round",
                label=names[scoring])
    ax.set_ylim(0, 1.02)
    ax.margins(x=0.02)
    _legend(ax, ncol=2)
    fig.tight_layout()
    if path is not None:
        save(fig, path)
    return fig


def save(fig: Figure, path) -> Path:
    """Save a figure; a bare filename goes into figures/ at the repo root."""
    path = Path(path)
    if not path.is_absolute() and path.parent == Path("."):
        path = FIGURES_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor())
    return path
