from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Robot:
    """Robot following a pre-planned path with failure handling and collision avoidance."""
    id: int
    path: List[Tuple[int, int]]
    idx: int = 0
    failed: bool = False

    @property
    def pos(self) -> Tuple[int, int]:
        """Current position on path."""
        return self.path[self.idx]
    
    @property
    def x(self) -> int:
        """Current x coordinate."""
        return self.path[self.idx][0]
    
    @property
    def y(self) -> int:
        """Current y coordinate."""
        return self.path[self.idx][1]

    def is_clear(self, nx: int, ny: int, all_robots: List['Robot'], radius: int = 1) -> bool:
        """
        Check if moving to (nx, ny) violates the collision radius of any OTHER robot.
        Radius 1 means: Do not enter the 3x3 zone centered on another robot.
        """
        for r in all_robots:
            if r.id == self.id:
                continue
            # Chebyshev distance (square neighborhood) check
            if max(abs(r.x - nx), abs(r.y - ny)) <= radius:
                return False
        return True

    def step(self, all_robots: List['Robot'] = None):
        """
        Advance one step along path (unless failed or blocked by collision).
        If the next position would collide with another robot, wait at current position.
        """
        if self.failed:
            return
        
        # Check if we can advance
        if self.idx < len(self.path) - 1:
            next_pos = self.path[self.idx + 1]
        
            self.idx += 1

