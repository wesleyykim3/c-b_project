"""Filler (distractor) material for Experiment 1.

Two independent knobs, varied orthogonally in the sweep:
  - filler_tokens: the TOTAL size of the filler budget (the "elapsed time"
    analogue).
  - filler_items: the number of DISCRETE filler chunks the budget is split into
    (the "number of intervening events" analogue, cf. Baddeley and Hitch).

Two filler TYPES:
  - "passive": neutral prose the model just reads.
  - "effortful": arithmetic problems the model is told to work through. This is
    the closer analogue, since human recency only broke under an effortful,
    paced distractor. Note the disanalogy: a single-prompt LLM has no real time
    axis, so "delay" here is tokens of required intervening material.

build_filler returns a LIST of `items` chunks whose combined length is about
`tokens` tokens. The condition builder decides where they go (one block after
the list for C1, spread across gaps for C2).
"""

from __future__ import annotations
import random

PASSIVE_SENTENCES = [
    "The weather stayed mild through the afternoon.",
    "A train passed somewhere beyond the hills.",
    "The kettle cooled slowly on the counter.",
    "Leaves drifted across the empty courtyard.",
    "The clock on the wall kept its steady pace.",
    "A pale light settled over the quiet street.",
    "The river moved without any hurry at all.",
    "Dust gathered along the edge of the shelf.",
    "The lamp flickered once and then held firm.",
    "Footsteps faded down the long stone hallway.",
]

# Rough token estimate: ~1.3 tokens per whitespace word. Good enough for sizing.
_TOK_PER_WORD = 1.3


def _approx_tokens(text: str) -> int:
    return max(1, round(len(text.split()) * _TOK_PER_WORD))


def _passive_chunk(target_tokens: int, rng: random.Random) -> str:
    out = []
    while _approx_tokens(" ".join(out)) < target_tokens:
        out.append(rng.choice(PASSIVE_SENTENCES))
    return " ".join(out)


def _effortful_chunk(target_tokens: int, rng: random.Random) -> str:
    # Each problem is ~10 tokens including the instruction scaffold.
    out = ["Solve each problem, then continue:"]
    while _approx_tokens(" ".join(out)) < target_tokens:
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        op = rng.choice(["+", "-"])
        out.append(f"What is {a} {op} {b}?")
    return " ".join(out)


def build_filler(filler_type: str, tokens: int, items: int,
                 seed: int) -> list[str]:
    """Return `items` chunks totaling about `tokens` tokens."""
    if items < 1:
        return []
    rng = random.Random(seed)
    per = max(1, tokens // items)
    make = _passive_chunk if filler_type == "passive" else _effortful_chunk
    if filler_type not in ("passive", "effortful"):
        raise ValueError(f"unknown filler_type {filler_type!r}")
    return [make(per, rng) for _ in range(items)]


if __name__ == "__main__":
    for ft in ("passive", "effortful"):
        chunks = build_filler(ft, tokens=120, items=3, seed=0)
        print(f"--- {ft}: {len(chunks)} chunks ---")
        for c in chunks:
            print(" ", c[:80], "...")
