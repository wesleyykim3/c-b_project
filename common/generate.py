"""Fact-list generation for the serial-position experiments.

A "list" is a sequence of facts of the form "<Name> <relation> <object>",
e.g. "Paul has a guitar". Each fact sits at a known serial position, which is
what the serial-position curve is computed over.

Targets are single words so scoring is a clean exact match. The object pool is
reused across facts on purpose, so items compete (interference) and the model
is less likely to hit the ceiling at every position.
"""

from __future__ import annotations
import random
from dataclasses import dataclass, asdict

# Pools. Names are unique within a list; objects/cities/professions may repeat
# across facts within a list, which increases interference.
# Deduped at import so an accidental repeat shrinks the pool instead of letting
# two facts share a name (which would make name-keyed scoring ambiguous).
NAMES = list(dict.fromkeys([
    "Paul", "Ann", "Mark", "Lucy", "John", "Emma", "Dave", "Nora", "Carl", "Ruth",
    "Sean", "Jane", "Luke", "Iris", "Adam", "Beth", "Noah", "Kate", "Finn", "Gail",
    "Jack", "Tess", "Cole", "Mila", "Ross", "Dana", "Seth", "Lena", "Wade", "Faye",
    "Reed", "Joan", "Troy", "Nina", "Glen", "Rita", "Chad", "Lois", "Dean", "Vera",
    "Gene", "Hope", "Kurt", "Faith", "Neil", "Joy", "Blake", "Rose", "Grant", "Lily",
    "Hugh", "Pearl", "Lloyd", "Opal", "Clark", "Maeve", "Brent", "Elsie", "Craig", "Greta",
    "Dale", "Ingrid", "Floyd", "Mabel", "Ralph", "Edith", "Roy", "Cora", "Vance", "Myra",
    "Scott", "Dawn", "Keith", "Jade", "Bruce", "Fern", "Lyle", "Hazel", "Cyrus", "Elle",
    "Marty", "Wren", "Rex", "Sage", "Dell", "Lark", "Gus", "Nell", "Rory", "Hank",
    "Otis", "Bess", "Ned", "Liv", "Abe", "Ivy", "Dirk", "Mae", "Zane", "Kirk",
    "Bea", "Wes", "Isla", "Theo", "Eve", "Sam", "Ben", "Tom", "Nan", "Jed",
    "Lon", "Deb", "Hal", "Gil", "Kay", "Zoe", "Dom", "Ada", "Max", "Pia",
]))

OBJECTS = {
    "has-a": [
        "guitar", "bicycle", "camera", "kettle", "ladder", "trumpet", "hammer",
        "compass", "lantern", "violin", "shovel", "anchor", "candle", "mirror",
    ],
    "lives-in": [
        "Dublin", "Oslo", "Cairo", "Lima", "Tokyo", "Perth", "Quito", "Rome",
        "Accra", "Riga", "Bern", "Dhaka", "Hanoi", "Minsk",
    ],
    "is-a": [
        "physicist", "baker", "sailor", "painter", "welder", "dentist", "farmer",
        "pilot", "chemist", "tailor", "plumber", "surgeon", "botanist", "mason",
    ],
}

RELATION_PHRASE = {
    "has-a": "has a",
    "lives-in": "lives in",
    "is-a": "is a",
}


@dataclass
class Fact:
    position: int      # 1-indexed serial position in the list
    name: str
    relation: str      # has-a | lives-in | is-a
    obj: str           # the single-word target
    text: str          # rendered sentence


def render(name: str, relation: str, obj: str) -> str:
    return f"{name} {RELATION_PHRASE[relation]} {obj}."


def generate_list(length: int, seed: int, template: str = "has-a",
                  confusable: bool = True) -> list[Fact]:
    """Return a list of `length` Facts.

    template: one of "has-a", "lives-in", "is-a".
    confusable=True draws objects WITH replacement from the pool, so the same
    object recurs across people in one list (more interference). False draws
    distinct objects where the pool allows.
    """
    if template not in OBJECTS:
        raise ValueError(f"unknown template {template!r}")
    rng = random.Random(seed)
    names = rng.sample(NAMES, length)  # unique names, fails loudly if length > pool
    pool = OBJECTS[template]
    if confusable:
        objs = [rng.choice(pool) for _ in range(length)]
    else:
        if length > len(pool):
            raise ValueError(
                f"length {length} exceeds {template} pool ({len(pool)}); "
                "use confusable=True or a shorter list")
        objs = rng.sample(pool, length)
    facts = []
    for i, (nm, ob) in enumerate(zip(names, objs), start=1):
        facts.append(Fact(i, nm, template, ob, render(nm, template, ob)))
    return facts


def facts_to_dicts(facts: list[Fact]) -> list[dict]:
    return [asdict(f) for f in facts]


if __name__ == "__main__":
    for f in generate_list(8, seed=1, template="has-a"):
        print(f.position, f.text)
