import numpy as np
from typing import List, Tuple
from ..common.geometry import manhattan_connect


# def sensor_aware_path_for_region(mask: np.ndarray) -> List[Tuple[int, int]]:
#     """Generate boustrophedon-style sweeping path for a region mask."""
#     H, W = mask.shape
#     path: List[Tuple[int, int]] = []
#     lr = True
#     started = False
#     for y in range(0, H, 2):
#         xs = [x for x in range(W) if mask[y, x]]
#         if not xs:
#             continue
#         segs = []
#         s = xs[0]
#         p = xs[0]
#         for x in xs[1:]:
#             if x == p + 1:
#                 p = x
#             else:
#                 segs.append((s, p))
#                 s = x
#                 p = x
#         segs.append((s, p))
#         segs = segs if lr else list(reversed(segs))
#         for (a, b) in segs:
#             run = list(range(a, b + 1)) if lr else list(range(b, a - 1, -1))
#             if not started:
#                 path.append((run[0], y))
#                 started = True
#             else:
#                 manhattan_connect(path, run[0], y)
#             for x in run[1:]:
#                 path.append((x, y))
#         lr = not lr
#         if y + 2 < H:
#             xs_next = [x for x in range(W) if mask[y + 2, x]]
#             if xs_next:
#                 x_curr, _ = path[-1]
#                 x_next = min(xs_next, key=lambda xx: abs(xx - x_curr))
#                 manhattan_connect(path, x_next, y + 2)
#     return path


def sensor_aware_path_for_region(mask: np.ndarray, robot_radius: int) -> List[Tuple[int, int]]:
    """
    Generate boustrophedon-style sweeping path for a region mask,
    adjusted for robot radius to minimize overlap.
    """
    H, W = mask.shape
    path: List[Tuple[int, int]] = []
    
    # Calculate stride: Diameter of the sensor. 
    # Using 2 * radius ensures the "top" of one sweep touches the "bottom" of the next.
    step_size = max(1, 2 * robot_radius)
    
    lr = True
    started = False
    
    for y in range(0, H, step_size):
        xs = [x for x in range(W) if mask[y, x]]
        
        if not xs:
            continue

        segs = []
        if xs:
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

        # Reverse segments if moving Right-to-Left
        segs = segs if lr else list(reversed(segs))

        for (a, b) in segs:
            # Generate the coordinates for this segment
            run = list(range(a, b + 1)) if lr else list(range(b, a - 1, -1))
            
            if not started:
                path.append((run[0], y))
                started = True
            else:
                manhattan_connect(path, run[0], y)
            
            for x in run[1:]:
                path.append((x, y))

        lr = not lr

        if y + step_size < H:
            xs_next = [x for x in range(W) if mask[y + step_size, x]]
            if xs_next:
                x_curr, _ = path[-1]
                x_next = min(xs_next, key=lambda xx: abs(xx - x_curr))
                
                manhattan_connect(path, x_next, y + step_size)

    return path