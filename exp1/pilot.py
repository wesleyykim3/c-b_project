"""Ceiling pilot: pick the two list lengths for Experiment 1.

Runs the no-filler ("none") free-recall task across several list lengths on ONE
cheap model and reports accuracy by serial position. You want lengths where
mid-list accuracy is clearly below 100 percent, because a length where the model
scores ~1.00 everywhere has no curve to measure (the ceiling problem).

This file carries a MINIMAL parser/scorer, enough to read the pilot. The
canonical scorer is Person 2's score.py. Do not import this one into the main
analysis.

From the serial_position/ directory:
  python -m exp1.pilot            # 20 lists per length
  python -m exp1.pilot --n 10     # quicker, cheaper
"""

from __future__ import annotations
import re
import argparse
from collections import defaultdict

from common.generate import generate_list
from common.call import call_model
from exp1.conditions import build_prompt

# One cheap model is enough to find the ceiling. A small model ceilings later,
# so Llama-3.2-3B is a good choice if you want a curve at shorter lengths; an
# 8B often stays near 100% on plain recall (see note at the bottom).
PILOT_MODEL = ("meta-llama/llama-3.2-3b-instruct",
               "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY")
LENGTHS = [40, 80, 100]          # keep each <= the NAMES pool in generate.py
N_LISTS = 20
TEMPERATURE = 0.0


def parse_recall(text: str) -> dict[str, str]:
    """Parse 'Name: object' lines into {name_lower: object_lower}.

    Tolerant of bullets, numbering, and dashes. Minimal on purpose.
    """
    out = {}
    for line in text.splitlines():
        m = re.match(r"\s*[-*\d.\)]*\s*([A-Za-z]+)\s*[:\-]\s*([A-Za-z]+)", line)
        if m:
            out[m.group(1).lower()] = m.group(2).lower()
    return out


def score(parsed: dict[str, str], facts) -> list[tuple[int, bool]]:
    """Per-fact (position, correct) using a name -> object match."""
    return [(f.position, parsed.get(f.name.lower()) == f.obj.lower())
            for f in facts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_LISTS,
                    help="lists per length")
    args = ap.parse_args()
    model, base_url, key_env = PILOT_MODEL

    print(f"pilot model: {model}   ({args.n} lists per length)\n")
    print(f"{'length':>6}  {'overall':>7}  {'mid-list':>8}  verdict")
    suggestions = []
    for length in LENGTHS:
        pos_correct = defaultdict(int)
        pos_total = defaultdict(int)
        for list_id in range(args.n):
            facts = generate_list(length, seed=list_id, template="has-a")
            prompt = build_prompt(facts, "none")
            r = call_model(model, prompt, temperature=TEMPERATURE,
                           max_tokens=length * 8 + 64,
                           base_url=base_url, api_key_env=key_env)
            for position, correct in score(parse_recall(r.text), facts):
                pos_total[position] += 1
                pos_correct[position] += int(correct)

        overall = sum(pos_correct.values()) / max(1, sum(pos_total.values()))
        lo, hi = length // 3, 2 * length // 3           # middle third
        mid = [pos_correct[p] / pos_total[p]
               for p in range(lo + 1, hi + 1) if pos_total[p]]
        mid_acc = sum(mid) / len(mid) if mid else float("nan")
        ceilinged = mid_acc >= 0.95
        if not ceilinged:
            suggestions.append(length)
        print(f"{length:>6}  {overall:>7.2f}  {mid_acc:>8.2f}  "
              f"{'ceilinged' if ceilinged else 'usable'}")

    print("\nPick two 'usable' lengths (mid-list accuracy below ~0.95); those")
    print("have a real curve. Lengths near 1.00 everywhere are ceilinged.")
    if len(suggestions) >= 2:
        print(f"Suggested: {suggestions[0]} and {suggestions[-1]}")
    elif suggestions:
        print(f"Only {suggestions[0]} looks usable; try longer lists or "
              "confusable=True / a harder template.")
    else:
        print("All lengths ceilinged. A strong instruct model can hold plain "
              "lists near-perfectly, so either:")
        print("  - go longer (raise lengths and the NAMES pool), or")
        print("  - pilot a smaller model (e.g. meta-llama/llama-3.2-3b-instruct), or")
        print("  - accept a ceilinged baseline and let the Experiment 1 filler "
              "conditions induce the curve (that is the actual manipulation).")


if __name__ == "__main__":
    main()
