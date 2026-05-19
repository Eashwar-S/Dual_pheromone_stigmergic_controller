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


def plot_visit_counts(
    visit_counts: np.ndarray,
    robots,
    targets,
    found_targets,
    output_path=None,
):
    """
    Plot continuous visit-count heatmap.
    More visits are shown as darker/redder cells.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    H, W = visit_counts.shape

    max_visits = int(visit_counts.max())

    fig, ax = plt.subplots(figsize=(8, 8))

    img = ax.imshow(
        visit_counts,
        origin="lower",
        cmap="Reds",
        norm=Normalize(vmin=0, vmax=max(1, max_visits)),
        extent=[0, W, 0, H],
    )

    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label("Number of visits")

    undiscovered = targets - found_targets
    discovered = found_targets

    if undiscovered:
        ux, uy = zip(*undiscovered)
        ax.scatter(
            [x + 0.5 for x in ux],
            [y + 0.5 for y in uy],
            s=25,
            marker="x",
            c="black",
            label="Undiscovered targets",
            zorder=3,
        )

    if discovered:
        dx, dy = zip(*discovered)
        ax.scatter(
            [x + 0.5 for x in dx],
            [y + 0.5 for y in dy],
            s=30,
            marker="o",
            facecolors="none",
            edgecolors="blue",
            label="Discovered targets",
            zorder=3,
        )

    robot_x = [r.x + 0.5 for r in robots]
    robot_y = [r.y + 0.5 for r in robots]

    ax.scatter(
        robot_x,
        robot_y,
        s=40,
        c="black",
        label="Final robot positions",
        zorder=4,
    )

    ax.set_title(f"Final Visit Count Heatmap | Max visits = {max_visits}")
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xticks(np.arange(0, W + 1, 10))
    ax.set_yticks(np.arange(0, H + 1, 10))
    ax.grid(which="major", color="k", alpha=0.15, linewidth=0.5)

    ax.legend(loc="upper right", fontsize=8)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Final visit-count heatmap saved to: {output_path}")

    plt.show()