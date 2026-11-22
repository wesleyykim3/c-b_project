"""Dual scorer.

Scores each trial two ways from the same parsed output:

  free_correct   : fact at position p is correct if its name is recalled with
                   the right object ANYWHERE in the output.
  serial_correct : correct only if recalled with the right object at output
                   rank p (strict serial order).

Rules :
  - Only the FIRST occurrence of each name is scored; later repeats never
    rescue an earlier wrong answer.
  - Rank counts first occurrences only, so loops can't shift later items.
    Intrusions and wrong-object entries occupy a rank (as in human strict
    serial scoring). No skip tolerance: one omission shifts every later rank.

score_row takes a raw trial dict and returns (per-position rows, per-trial
summary), carrying all trial metadata through unchanged.
"""

from __future__ import annotations
from common.parse import parse_recall

# Raw-row keys that are payload, not metadata.
_PAYLOAD = ("facts", "raw_output", "cached")


def _meta(row: dict) -> dict:
    return {k: v for k, v in row.items() if k not in _PAYLOAD}


def score_row(row: dict) -> tuple[list[dict], dict]:
    facts = row["facts"]
    parsed = parse_recall(row["raw_output"], facts)
    meta = _meta(row)

    # First occurrences only, ranked in output order.
    firsts = [e for e in parsed.events if not e.repeat]
    first_by_pos: dict[int, tuple[int, object]] = {}   # position -> (rank, event)
    for rank, e in enumerate(firsts, start=1):
        if e.matched_position is not None:
            first_by_pos[e.matched_position] = (rank, e)

    position_rows = []
    for f in facts:
        pos = f["position"] if isinstance(f, dict) else f.position
        hit = first_by_pos.get(pos)
        rank, ev = hit if hit else (None, None)
        position_rows.append({
            **meta,
            "position": pos,
            "free_correct": bool(ev and ev.correct),
            "serial_correct": bool(ev and ev.correct and rank == pos),
            "recalled_rank": rank,
        })

    n_facts = len(facts)
    summary = {
        **meta,
        "n_facts": n_facts,
        "n_events": len(parsed.events),
        "parse_rate": parsed.parse_rate,
        "free_acc": sum(r["free_correct"] for r in position_rows) / n_facts,
        "serial_acc": sum(r["serial_correct"] for r in position_rows) / n_facts,
        "intrusions": sum(e.matched_position is None for e in firsts),
        "wrong_object": sum(e.matched_position is not None and not e.correct
                            for e in firsts),
        "repeats": sum(e.repeat for e in parsed.events),
        "omissions": n_facts - len(first_by_pos),   # facts never recalled by name
    }
    return position_rows, summary


def score_rows(rows) -> tuple[list[dict], list[dict]]:
    """Score many raw trials. Returns (all position rows, all trial summaries)."""
    pos_rows, summaries = [], []
    for row in rows:
        p, s = score_row(row)
        pos_rows.extend(p)
        summaries.append(s)
    return pos_rows, summaries
