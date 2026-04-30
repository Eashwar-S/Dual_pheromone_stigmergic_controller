import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from pathlib import Path
from typing import Tuple, Dict


def coverage_to_image(covered_bool: np.ndarray) -> np.ndarray:
    """Convert boolean coverage grid to grayscale image."""
    img = np.ones(covered_bool.shape, dtype=float)
    img[covered_bool] = 0.85
    return img


def pheromone_to_rgba(ph: np.ndarray, alpha_scale: float = 0.35, pher_min: float = 1e-6) -> np.ndarray:
    """Convert pheromone field to RGBA image (pink/magenta)."""
    vmax = max(np.percentile(ph, 95), pher_min)
    norm = np.clip(ph / vmax, 0.0, 1.0)
    rgba = np.zeros((ph.shape[0], ph.shape[1], 4), dtype=float)
    rgba[..., 0] = 1.0
    rgba[..., 1] = 0.2
    rgba[..., 2] = 0.6
    rgba[..., 3] = norm * alpha_scale
    return rgba


def combined_pheromone_to_rgba(p_rep: np.ndarray, p_attr: np.ndarray) -> np.ndarray:
    """Combine repulsive (pink) and attractive (blue) pheromone fields into RGBA."""
    H, W = p_rep.shape
    rgba = np.zeros((H, W, 4), dtype=float)
    
    max_rep = np.percentile(p_rep, 98) if np.max(p_rep) > 0 else 1.0
    norm_rep = np.clip(p_rep / (max_rep + 1e-9), 0, 1) * 0.5
    norm_attr = np.clip(p_attr / 10.0, 0, 1) * 0.9
    
    rgba[..., 0] = norm_rep * 1.0
    rgba[..., 1] = norm_rep * 0.2 + norm_attr * 0.2
    rgba[..., 2] = norm_rep * 0.6 + norm_attr * 1.0
    rgba[..., 3] = np.clip(norm_rep + norm_attr, 0, 1)
    
    return np.clip(rgba, 0.0, 1.0)


def region_colors(zones: np.ndarray, alpha: float = 0.12) -> Tuple[np.ndarray, Dict]:
    """Generate colored RGBA overlay for Voronoi regions."""
    cmap = get_cmap('tab20', np.max(zones) + 1)
    rgba = cmap(zones)
    rgba[..., 3] = alpha
    return rgba, {i: cmap(i) for i in np.unique(zones)}


def draw_voronoi_borders(ax: plt.Axes, zones: np.ndarray, color: str = '#0b3d91', 
                         lw: float = 1.2, alpha: float = 0.9):
    """Draw borders along edges where adjacent cells belong to different zones."""
    H, W = zones.shape
    for y in range(H):
        x = 0
        while x < W - 1:
            if zones[y, x] != zones[y, x + 1]:
                x0 = x + 1
                ax.plot([x0, x0], [y, y + 1], color=color, linewidth=lw, alpha=alpha, zorder=3)
            x += 1
    for x in range(W):
        y = 0
        while y < H - 1:
            if zones[y, x] != zones[y + 1, x]:
                y0 = y + 1
                ax.plot([x, x + 1], [y0, y0], color=color, linewidth=lw, alpha=alpha, zorder=3)
            y += 1


def save_targets_over_time_plot(path: Path, targets_found_over_time: list):
    """Save dotted line plot of targets detected over time."""
    xs = np.arange(len(targets_found_over_time))
    ys = np.asarray(targets_found_over_time, dtype=float)
    
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(xs, ys, linestyle=':', linewidth=2)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Total targets detected")
    ax.set_title("Targets detected over time")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
