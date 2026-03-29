from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

BACKGROUND_NAME = "__background__"
BACKGROUND_ID = 0


@dataclass
class ClassMap:
    name_to_id: Dict[str, int]

    @property
    def id_to_name(self) -> Dict[int, str]:
        return {idx: name for name, idx in self.name_to_id.items()}

    def num_classes(self) -> int:
        return len(self.name_to_id)

    def to_serializable(self) -> Dict[str, int]:
        return dict(self.name_to_id)

    @classmethod
    def from_names(cls, names: List[str]) -> "ClassMap":
        mapping = {BACKGROUND_NAME: BACKGROUND_ID}
        for offset, name in enumerate(sorted(set(names)), start=1):
            if name == BACKGROUND_NAME:
                continue
            mapping[name] = offset
        return cls(mapping)
