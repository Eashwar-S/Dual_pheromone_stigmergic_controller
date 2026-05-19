from typing import List, Tuple


def manhattan_path(start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Create a Manhattan path from start to end (horizontal first, then vertical)."""
    path = [start]
    x1, y1 = start
    x2, y2 = end
    
    dx = 1 if x2 > x1 else -1
    while x1 != x2:
        x1 += dx
        path.append((x1, y1))
    
    dy = 1 if y2 > y1 else -1
    while y1 != y2:
        y1 += dy
        path.append((x1, y1))
    
    return path


def manhattan_connect(path: List[Tuple[int, int]], x2: int, y2: int):
    """Append Manhattan path from last position in path to (x2, y2)."""
    if not path:
        return
    x1, y1 = path[-1]
    dx = 1 if x2 > x1 else -1
    while x1 != x2:
        x1 += dx
        path.append((x1, y1))
    dy = 1 if y2 > y1 else -1
    while y1 != y2:
        y1 += dy
        path.append((x1, y1))
