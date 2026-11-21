"""Experiment 1 driver: delayed vs continual distractors.

Writes one RAW trial per line to data/exp1_raw.jsonl. Scoring (Person 2's
score.py) turns each raw trial into the per-position schema rows. Keeping run
and scoring separate means you can re-score without re-calling the models.

Cells:
  C0 "none"       : one cell per list length (no filler).
  C1 "after_list" } grid of filler_type x filler_tokens x filler_items
  C2 "continual"  } at each list length (default: effortful x {500,1500} x {1,6}).

From the serial_position/ directory:
  python -m exp1.run_exp1 --limit 2   (smoke test)
  python -m exp1.run_exp1              (full)
Use --limit to run only the first N lists per cell while testing.
"""

from __future__ import annotations
import json
import argparse
import itertools
from pathlib import Path

from common.generate import generate_list, facts_to_dicts
from common.call import call_model
from exp1.conditions import build_prompt

OUT = Path(__file__).resolve().parent.parent / "data" / "exp1_raw.jsonl"

# ---- CONFIG: set these with your group ----------------------------------
CONFIG = {
    # OpenRouter models: (model_string, base_url, api_key_env).
    # Smoke/pilot: 70B and gpt-4o-mini stay ceilinged even under fillers; the
    # distractor dissociation shows up on the small Llama ladder (3B -> 8B).
    # Re-add a big model later only as a thin ceiling check, not the full grid.
    # Verify ids: curl -s https://openrouter.ai/api/v1/models
    "models": [
        ("meta-llama/llama-3.2-3b-instruct", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
        ("meta-llama/llama-3.1-8b-instruct", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    ],
    "template": "has-a",
    # Baseline "none" may stay near ceiling; the DV is C1 vs C2 under effortful
    # filler. 40/80 give room for last-k recency without exhausting the name pool.
    "lengths": [40, 80],
    # Effortful first (human-valid analogue). Add "passive" in a second pass
    # if budget allows a type dissociation.
    "filler_types": ["effortful"],
    "filler_tokens": [500, 1500],   # "elapsed tokens" analogue
    "filler_items": [1, 6],        # few vs many intervening events
    "n_lists": 50,
    "temperature": 0.0,
    "max_tokens": 1024,            # room for L=80 "Name: object" lines
}
# -------------------------------------------------------------------------


def cells(cfg):
    """Yield (condition, length, filler_type, filler_tokens, filler_items)."""
    for length in cfg["lengths"]:
        yield ("none", length, None, 0, 0)
    for condition in ("after_list", "continual"):
        for length, ft, tok, it in itertools.product(
                cfg["lengths"], cfg["filler_types"],
                cfg["filler_tokens"], cfg["filler_items"]):
            yield (condition, length, ft, tok, it)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="only run the first N lists per cell (smoke test)")
    args = ap.parse_args()
    cfg = CONFIG
    OUT.parent.mkdir(exist_ok=True)

    n = args.limit or cfg["n_lists"]
    written = 0
    with OUT.open("a") as fh:
        for (model, base_url, key_env) in cfg["models"]:
            for (cond, length, ftype, ftok, fit) in cells(cfg):
                for list_id in range(n):
                    facts = generate_list(length, seed=list_id,
                                          template=cfg["template"])
                    prompt = build_prompt(
                        facts, cond, filler_type=ftype or "passive",
                        filler_tokens=ftok, filler_items=fit, seed=list_id)
                    r = call_model(model, prompt,
                                   temperature=cfg["temperature"],
                                   max_tokens=cfg["max_tokens"],
                                   base_url=base_url, api_key_env=key_env)
                    row = {
                        "model": model, "list_id": list_id,
                        "list_length": length, "condition": cond,
                        "filler_type": ftype, "filler_tokens": ftok,
                        "filler_items": fit, "temperature": cfg["temperature"],
                        "facts": facts_to_dicts(facts),
                        "raw_output": r.text, "cached": r.cached,
                    }
                    fh.write(json.dumps(row) + "\n")
                    written += 1
    print(f"wrote {written} raw trials to {OUT}")


if __name__ == "__main__":
    main()
