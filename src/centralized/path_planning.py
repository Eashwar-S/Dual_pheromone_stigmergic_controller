import numpy as np
from typing import List, Tuple
from ..common.geometry import manhattan_connect


def sensor_aware_path_for_region(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Generate boustrophedon-style sweeping path for a region mask."""
    H, W = mask.shape
    path: List[Tuple[int, int]] = []
    lr = True
    started = False
    for y in range(0, H, 2):
        xs = [x for x in range(W) if mask[y, x]]
        if not xs:
            continue
        segs = []
        s = xs[0]
        p = xs[0]
        for x in xs[1:]:
            if x == p + 1:
                p = x
            else:
                segs.append((s, p))
                s = x
                p = x
        segs.append((s, p))
        segs = segs if lr else list(reversed(segs))
        for (a, b) in segs:
            run = list(range(a, b + 1)) if lr else list(range(b, a - 1, -1))
            if not started:
                path.append((run[0], y))
                started = True
            else:
                manhattan_connect(path, run[0], y)
            for x in run[1:]:
                path.append((x, y))
        lr = not lr
        if y + 2 < H:
            xs_next = [x for x in range(W) if mask[y + 2, x]]
            if xs_next:
                x_curr, _ = path[-1]
                x_next = min(xs_next, key=lambda xx: abs(xx - x_curr))
                manhattan_connect(path, x_next, y + 2)
    return path
