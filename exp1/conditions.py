"""Prompt construction for the three Experiment 1 conditions.

Conditions (where the filler goes):
  C0 "none"      : facts, then recall. The baseline curve.
  C1 "after_list": all facts, then ONE filler block, then recall. The
                   Glanzer and Cunitz filled-delay analogue.
  C2 "continual" : a filler chunk after EVERY fact, then recall. The
                   continuous-distractor analogue.

The filler budget (filler_tokens) and chunk count (filler_items) are set by the
caller and are identical across C1 and C2 for a given cell, so the ONLY thing
that differs between C1 and C2 is placement. That is what dissociates a
short-term-store account (recency dies in C1, survives C2) from a context
account.

Recall instruction asks for "Name: object" lines so parsing is trivial.
"""

from __future__ import annotations
from exp1.filler import build_filler
from common.generate import Fact

RECALL_INSTRUCTION = (
    "Now recall every fact from the list above. "
    "Write one per line in the exact form 'Name: object'. "
    "List as many as you can, in any order."
)

# Effortful fillers ask the model to solve arithmetic mid-prompt; without this,
# small models often dump answers instead of facts (parse rate collapses to 0).
RECALL_INSTRUCTION_EFFORTFUL = (
    "Do not write answers to the arithmetic problems. "
    "Now recall every fact from the list above. "
    "Write one per line in the exact form 'Name: object'. "
    "List as many as you can, in any order."
)

CONDITIONS = ("none", "after_list", "continual")


def _recall_instruction(filler_type: str, condition: str) -> str:
    if condition != "none" and filler_type == "effortful":
        return RECALL_INSTRUCTION_EFFORTFUL
    return RECALL_INSTRUCTION


def build_prompt(facts: list[Fact], condition: str, filler_type: str = "passive",
                 filler_tokens: int = 0, filler_items: int = 0,
                 seed: int = 0) -> str:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")

    header = "Read and remember the following facts.\n"
    recall = _recall_instruction(filler_type, condition)

    if condition == "none":
        body = "\n".join(f.text for f in facts)
        return f"{header}{body}\n\n{recall}"

    chunks = build_filler(filler_type, filler_tokens, filler_items, seed) \
        if filler_items > 0 else []

    if condition == "after_list":
        body = "\n".join(f.text for f in facts)
        block = ("\n\n" + "\n".join(chunks) + "\n") if chunks else "\n"
        return f"{header}{body}\n{block}\n{recall}"

    # continual: distribute chunks across the gaps after each fact, round-robin.
    lines = []
    for i, f in enumerate(facts):
        lines.append(f.text)
        # place chunk j after fact j, wrapping if items > length
        for j, ch in enumerate(chunks):
            if j % len(facts) == i:
                lines.append(ch)
    body = "\n".join(lines)
    return f"{header}{body}\n\n{recall}"


if __name__ == "__main__":
    from common.generate import generate_list
    fs = generate_list(4, seed=1)
    for cond in CONDITIONS:
        print(f"\n===== {cond} =====")
        print(build_prompt(fs, cond, "effortful", filler_tokens=40,
                           filler_items=2, seed=7))
