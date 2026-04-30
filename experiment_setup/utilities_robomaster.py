import numpy as np
from typing import Set, Tuple, List


def generate_unique_targets(grid_size: int, m: int, rng: np.random.Generator) -> Set[Tuple[int, int]]:
    """Generate m unique random target positions on a grid."""
    cells = [(x, y) for x in range(grid_size) for y in range(grid_size)]
    choices = rng.choice(len(cells), size=m, replace=False)
    return set(cells[i] for i in choices)


def neighbors_directional(x: int, y: int, heading: int, W: int, H: int, r: int) -> List[Tuple[int, int]]:
    """Return all cells within directional FOV (Manhattan distance r) from (x, y)."""
    out = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if abs(dx) + abs(dy) <= r:
                if heading == 0 and dy > 0: continue     # Up
                if heading == 90 and dx < 0: continue    # Right
                if heading == 180 and dy < 0: continue   # Down
                if heading == 270 and dx > 0: continue   # Left
                
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H:
                    out.append((nx, ny))
    return out


def mark_visible(grid_bool: np.ndarray, x: int, y: int, heading: int, r: int):
    """Mark all cells within directional FOV from (x, y). Increments int arrays, sets bool arrays to True."""
    H, W = grid_bool.shape
    for (nx, ny) in neighbors_directional(x, y, heading, W, H, r):
        if grid_bool.dtype == np.int_ or grid_bool.dtype == np.int32 or grid_bool.dtype == np.int64:
            grid_bool[ny, nx] += 1
        else:
            grid_bool[ny, nx] = True


def discover_targets_in_vnhood(x: int, y: int, heading: int, targets: Set[Tuple[int, int]], 
                                found: Set[Tuple[int, int]], W: int, H: int, r: int):
    """Discover targets within directional FOV and add to found set."""
    for (nx, ny) in neighbors_directional(x, y, heading, W, H, r):
        if (nx, ny) in targets:
            found.add((nx, ny))
