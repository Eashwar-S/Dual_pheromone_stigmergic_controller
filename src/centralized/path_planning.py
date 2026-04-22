import numpy as np
from typing import List, Tuple
from collections import deque


def _bfs_path_in_mask(mask: np.ndarray,
                      start: Tuple[int, int],
                      goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Shortest 4-neighbor path start->goal staying inside mask. [] if unreachable."""
    if start == goal:
        return [start]
    H, W = mask.shape
    sx, sy = start
    gx, gy = goal
    if not (0 <= sx < W and 0 <= sy < H and 0 <= gx < W and 0 <= gy < H):
        return []
    if not mask[sy, sx] or not mask[gy, gx]:
        return []

    q = deque([start])
    prev = {start: None}
    nbrs = [(1,0), (-1,0), (0,1), (0,-1)]

    while q:
        x, y = q.popleft()
        if (x, y) == goal:
            break
        for dx, dy in nbrs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H and mask[ny, nx] and (nx, ny) not in prev:
                prev[(nx, ny)] = (x, y)
                q.append((nx, ny))

    if goal not in prev:
        return []

    # reconstruct
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def sensor_aware_path_for_region(mask: np.ndarray, robot_radius: int) -> List[Tuple[int, int]]:
    """
    Generate boustrophedon-style sweeping path for a region mask.

    Fixes:
      A) Prevent leaving partition by using BFS connections that stay inside mask.
      B) Prevent coverage gaps by using overlapping stride (2R - 1).
      C) Start from a corner of the region bounding box (snapped into mask if needed).
    """
    H, W = mask.shape
    path: List[Tuple[int, int]] = []

    # --- FIX for gaps: overlapping stride (NOT 2R-1) ---
    step_size = max(1, 2 * robot_radius - 1)

    ys, xs = np.where(mask)
    if xs.size == 0:
        return path

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())

    # Choose bounding-box corner (bottom-left). Snap to nearest mask cell if corner not inside.
    desired_corner = (x_min, y_min)
    if not mask[desired_corner[1], desired_corner[0]]:
        d = np.abs(xs - desired_corner[0]) + np.abs(ys - desired_corner[1])
        k = int(np.argmin(d))
        start_corner = (int(xs[k]), int(ys[k]))
    else:
        start_corner = desired_corner

    def connect_inside_mask(to_xy: Tuple[int, int]) -> bool:
        """Connect current end of path to to_xy using BFS within mask."""
        nonlocal path
        if not path:
            path.append(to_xy)
            return True
        a = path[-1]
        bfs = _bfs_path_in_mask(mask, a, to_xy)
        if not bfs:
            return False
        # append without duplicating start node
        path.extend(bfs[1:])
        return True

    # Initialize at corner (guaranteed inside mask due to snap)
    path.append(start_corner)

    lr = True
    y0 = y_min

    while y0 <= y_max:
        y1 = min(y0 + step_size - 1, y_max)

        # pick a sweep row within band that intersects the mask
        sweep_y = None
        for yy in range(y0, y1 + 1):
            if mask[yy, :].any():
                sweep_y = yy
                break
        if sweep_y is None:
            y0 = y1 + 1
            continue

        xs_row = np.where(mask[sweep_y, :])[0].tolist()
        if not xs_row:
            y0 = y1 + 1
            continue

        # contiguous segments along that row
        segs = []
        s = xs_row[0]
        p = xs_row[0]
        for x in xs_row[1:]:
            if x == p + 1:
                p = x
            else:
                segs.append((s, p))
                s = x
                p = x
        segs.append((s, p))

        if not lr:
            segs = list(reversed(segs))

        for (a, b) in segs:
            run = list(range(a, b + 1)) if lr else list(range(b, a - 1, -1))
            # connect to start of this run INSIDE MASK
            ok = connect_inside_mask((run[0], sweep_y))
            if not ok:
                # If unreachable due to disconnected mask, skip this segment
                continue

            # emit points; each must be inside mask by construction
            for x in run[1:]:
                if mask[sweep_y, x]:
                    path.append((x, sweep_y))

        lr = not lr

        # connect to next band's first valid row in-mask
        next_y0 = y1 + 1
        if next_y0 <= y_max:
            next_y1 = min(next_y0 + step_size - 1, y_max)
            next_sweep_y = None
            for yy in range(next_y0, next_y1 + 1):
                if mask[yy, :].any():
                    next_sweep_y = yy
                    break
            if next_sweep_y is not None:
                xs_next = np.where(mask[next_sweep_y, :])[0].tolist()
                if xs_next:
                    x_curr, _ = path[-1]
                    x_next = min(xs_next, key=lambda xx: abs(xx - x_curr))
                    connect_inside_mask((x_next, next_sweep_y))

        y0 = y1 + 1

    return path