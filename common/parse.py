"""Recall-output parser.

Turns raw model text into an ORDERED list of recall events, keeping order,
repeats and intrusions (needed for PFR, lag-CRP and strict serial scoring).
This is the canonical parser; exp1/pilot.py's dict-based one drops order.

A line is parsed only if the WHOLE line is a recall entry:

    [bullet/number] Name <: or dash> object [trailing . , ;]

Bullets, markdown emphasis and case are tolerated. Prose, arithmetic answers
and entries with extra trailing text are skipped; they only lower parse_rate,
so format collapse shows up instead of creating false intrusions.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field

# Optional bullet or list number, then Name, a ':' or dash separator, then a
# single-word object, then optional trailing punctuation.
_LINE = re.compile(
    r"^\s*(?:[-*•]+|\d+[.)])?\s*"      # bullet / numbering
    r"([A-Za-z]+)"                      # name
    r"\s*(?::|[-–—])\s*"                # separator
    r"([A-Za-z]+)"                      # object (single word)
    r"\s*[.,;]?\s*$"                    # trailing punctuation only
)
_MARKDOWN = re.compile(r"[*_`]+")


@dataclass
class RecallEvent:
    rank: int                     # 1-indexed order in the output (parsed lines only)
    name: str                     # lowercased
    obj: str                      # lowercased
    matched_position: int | None  # serial position of the fact with this name; None = intrusion
    correct: bool                 # name known AND object matches that fact
    repeat: bool                  # name already recalled earlier in this output


@dataclass
class ParsedRecall:
    events: list[RecallEvent] = field(default_factory=list)
    n_lines: int = 0              # non-empty lines in the raw text
    n_parsed: int = 0             # lines that parsed as recall entries

    @property
    def parse_rate(self) -> float:
        """Parsed lines / non-empty lines (0.0 for empty output)."""
        return self.n_parsed / self.n_lines if self.n_lines else 0.0


def _get(fact, attr):
    """Facts may be common.generate.Fact objects or dicts from the raw JSONL."""
    return fact[attr] if isinstance(fact, dict) else getattr(fact, attr)


def parse_recall(text: str, facts) -> ParsedRecall:
    by_name = {_get(f, "name").lower(): f for f in facts}
    out = ParsedRecall()
    seen: set[str] = set()
    for raw in (text or "").splitlines():
        line = _MARKDOWN.sub("", raw).strip()
        if not line:
            continue
        out.n_lines += 1
        m = _LINE.match(line)
        if not m:
            continue
        name, obj = m.group(1).lower(), m.group(2).lower()
        fact = by_name.get(name)
        out.n_parsed += 1
        out.events.append(RecallEvent(
            rank=len(out.events) + 1,
            name=name,
            obj=obj,
            matched_position=_get(fact, "position") if fact is not None else None,
            correct=fact is not None and _get(fact, "obj").lower() == obj,
            repeat=name in seen,
        ))
        seen.add(name)
    return out
