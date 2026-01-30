from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Robot:
    """Robot following a pre-planned path with failure handling."""
    id: int
    path: List[Tuple[int, int]]
    idx: int = 0
    failed: bool = False

    @property
    def pos(self) -> Tuple[int, int]:
        """Current position on path."""
        return self.path[self.idx]

    def step(self):
        """Advance one step along path (unless failed)."""
        if self.failed:
            return
        if self.idx < len(self.path) - 1:
            self.idx += 1
