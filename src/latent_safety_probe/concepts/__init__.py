"""Contrastive concept sets."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Literal

import yaml

Role = Literal["safety", "control"]


@dataclass(frozen=True)
class Concept:
    name: str
    description: str
    role: Role
    positive: list[str]
    negative: list[str]

    @property
    def n_pairs(self) -> int:
        return len(self.positive)


@dataclass
class ConceptSet:
    concepts: dict[str, Concept] = field(default_factory=dict)

    def __iter__(self):
        return iter(self.concepts.values())

    def __getitem__(self, name: str) -> Concept:
        return self.concepts[name]

    @property
    def safety(self) -> list[Concept]:
        return [c for c in self if c.role == "safety"]

    @property
    def controls(self) -> list[Concept]:
        return [c for c in self if c.role == "control"]


def _parse(data: dict, source: str) -> ConceptSet:
    if not isinstance(data, dict):
        raise ValueError(f"{source}: root must be a mapping of concept name -> spec")
    out = ConceptSet()
    for name, spec in data.items():
        if not isinstance(spec, dict) or "pairs" not in spec:
            raise ValueError(f"{source}: concept '{name}' needs a 'pairs' list")
        role = spec.get("role", "safety")
        if role not in ("safety", "control"):
            raise ValueError(f"{source}: concept '{name}' role must be 'safety' or 'control'")
        pos, neg = [], []
        for i, p in enumerate(spec["pairs"]):
            if not isinstance(p, dict) or "positive" not in p or "negative" not in p:
                raise ValueError(f"{source}: concept '{name}' pair {i} needs positive+negative")
            pos.append(str(p["positive"]))
            neg.append(str(p["negative"]))
        if len(pos) < 4:
            raise ValueError(f"{source}: concept '{name}' needs at least 4 pairs (has {len(pos)})")
        out.concepts[name] = Concept(
            name=name,
            description=str(spec.get("description", "")),
            role=role,
            positive=pos,
            negative=neg,
        )
    return out


def load_default() -> ConceptSet:
    text = resources.files(__package__).joinpath("default.yaml").read_text(encoding="utf-8")
    return _parse(yaml.safe_load(text), "default.yaml")


def load_file(path: str | Path) -> ConceptSet:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        return _parse(yaml.safe_load(f), str(path))
