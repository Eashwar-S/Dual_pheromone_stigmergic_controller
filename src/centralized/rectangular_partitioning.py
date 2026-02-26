import numpy as np
from typing import Tuple, List

def rectangular_bisection_partition(grid_size: int, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Divide a square grid of size grid_size x grid_size into k rectangular regions.
    Returns:
        labels: (grid_size, grid_size) array of zone IDs.
        centers: (k, 2) array of geometric centers (x, y) for each zone.
    """
    labels = np.zeros((grid_size, grid_size), dtype=int)
    
    # Each item: (x_min, x_max, y_min, y_max, count_to_split)
    # x_max, y_max are inclusive
    rects = []
    
    def recursive_split(x_min, x_max, y_min, y_max, n):
        if n == 1:
            rects.append((x_min, x_max, y_min, y_max))
            return
        
        # Split n into n1 and n2
        n1 = n // 2
        n2 = n - n1
        
        w = x_max - x_min + 1
        h = y_max - y_min + 1
        
        if w >= h:
            # Split horizontally (vertical line)
            # Find split point such that area ratio is n1:n2
            # Total cells = w * h
            # Target cells for n1 = (n1 / n) * total_cells
            # x_split = x_min + round(w * n1 / n) - 1
            split_idx = int(round(w * n1 / n))
            split_idx = max(1, min(w - 1, split_idx))
            x_split = x_min + split_idx - 1
            
            recursive_split(x_min, x_split, y_min, y_max, n1)
            recursive_split(x_split + 1, x_max, y_min, y_max, n2)
        else:
            # Split vertically (horizontal line)
            split_idx = int(round(h * n1 / n))
            split_idx = max(1, min(h - 1, split_idx))
            y_split = y_min + split_idx - 1
            
            recursive_split(x_min, x_max, y_min, y_split, n1)
            recursive_split(x_min, x_max, y_split + 1, y_max, n2)

    recursive_split(0, grid_size - 1, 0, grid_size - 1, k)
    
    centers = np.zeros((k, 2))
    for i, (xmin, xmax, ymin, ymax) in enumerate(rects):
        labels[ymin:ymax+1, xmin:xmax+1] = i
        centers[i] = ((xmin + xmax) / 2.0 + 0.5, (ymin + ymax) / 2.0 + 0.5)
        
    return labels, centers
