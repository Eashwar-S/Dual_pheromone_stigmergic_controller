from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerPatch
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch


GRID_SIZE = 100
TARGET = np.array([50.0, 50.0])
FOLLOWER_START = np.array([85.0, 85.0])
OUTPUT_PATH = Path("convergence_spiral_gradient.png")
PHEROMONE_CORE_RADIUS = 7.0
PHEROMONE_MID_RADIUS = 16.0
PHEROMONE_FADE_RADIUS = 30.0


class HandlerArrow(HandlerPatch):
    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        arrow = FancyArrowPatch(
            (xdescent, ydescent + 0.5 * height),
            (xdescent + width, ydescent + 0.5 * height),
            arrowstyle="-|>",
            mutation_scale=fontsize * 1.45,
            linewidth=orig_handle.get_linewidth(),
            linestyle=orig_handle.get_linestyle(),
            color=orig_handle.get_edgecolor(),
        )
        arrow.set_transform(trans)
        return [arrow]


def make_attractive_pheromone(grid_size=GRID_SIZE, target=TARGET):
    """Create a blocky attractive field with a tight high-intensity target core."""
    axis = np.arange(grid_size)
    x, y = np.meshgrid(axis, axis)

    dx = np.abs(x - target[0])
    dy = np.abs(y - target[1])
    chebyshev_distance = np.maximum(dx, dy)
    euclidean_distance = np.hypot(dx, dy)

    core = np.exp(-((chebyshev_distance / PHEROMONE_CORE_RADIUS) ** 3.0))
    mid_field = np.exp(-((chebyshev_distance / PHEROMONE_MID_RADIUS) ** 2.5))
    outer_fade = np.exp(-((euclidean_distance / PHEROMONE_FADE_RADIUS) ** 3.1))
    pheromone = 0.02 + 0.98 * (0.42 * mid_field + 0.38 * outer_fade + 0.20 * core)
    pheromone = pheromone**1.45

    return np.clip(pheromone, 0.0, 1.0)


def make_square_spiral(center=TARGET, first_radius=5, spacing=5, max_radius=PHEROMONE_FADE_RADIUS):
    """Return a sparse square spiral path that stops around the visible field."""
    x, y = center
    points = [(x, y)]
    radius = first_radius

    while radius <= max_radius:
        points.extend(
            [
                (x + radius, y),
                (x + radius, y + radius),
                (x - radius, y + radius),
                (x - radius, y - radius),
            ]
        )
        if radius + spacing <= max_radius:
            points.append((x + radius + spacing, y - radius))
        radius += spacing

    path = np.array(points, dtype=float)
    return np.clip(path, 0, GRID_SIZE)


def make_gradient_path(start=FOLLOWER_START, target=TARGET):
    """A stylized follower trajectory descending the pheromone gradient."""
    return np.array(
        [
            start,
            (78, 83),
            (74, 78),
            (70, 74),
            (67, 69),
            (62, 66),
            (58, 61),
            (55, 57),
            target,
        ],
        dtype=float,
    )


def draw_gradient_arrows(ax, path):
    """Draw triangular arrowheads along the follower path."""
    for start, end in zip(path[:-1], path[1:]):
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(
                arrowstyle="-|>",
                color="black",
                linewidth=1.6,
                mutation_scale=26,
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=6,
        )


def style_grid_axis(ax):
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.arange(0, GRID_SIZE + 1, 20))
    ax.set_yticks(np.arange(0, GRID_SIZE + 1, 20))
    ax.grid(True, linestyle=":", linewidth=0.8, color="#aeb7bf", alpha=0.65)
    ax.tick_params(labelsize=12)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#222222")


def plot_convergence_spiral_gradient(output_path=OUTPUT_PATH):
    pheromone = make_attractive_pheromone()
    spiral_path = make_square_spiral()
    gradient_path = make_gradient_path()

    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 13,
            "legend.fontsize": 12,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.4), dpi=160, constrained_layout=True)

    for ax in axes:
        ax.imshow(
            pheromone,
            cmap="Blues",
            origin="lower",
            extent=(0, GRID_SIZE, 0, GRID_SIZE),
            interpolation="nearest",
            alpha=0.86,
            zorder=0,
        )
        style_grid_axis(ax)

    # axes[0].set_title("ADVERTISE Behavior")
    axes[0].set_xlabel("W")
    axes[0].set_ylabel("H")
    axes[0].plot(
        spiral_path[:, 0],
        spiral_path[:, 1],
        color="#8b008b",
        linewidth=1.1,
        zorder=4,
    )
    axes[0].plot(
        TARGET[0],
        TARGET[1],
        marker="x",
        color="red",
        markersize=11,
        markeredgewidth=3,
        linestyle="None",
        zorder=7,
    )

    arrow_indices = [2, 7, 12, 17, 22, 27, 32, 37]
    for index in arrow_indices:
        if index + 1 >= len(spiral_path):
            continue
        axes[0].annotate(
            "",
            xy=spiral_path[index + 1],
            xytext=spiral_path[index],
            arrowprops=dict(
                arrowstyle="->",
                color="#8b008b",
                linewidth=1.35,
                mutation_scale=16,
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=5,
        )

    axes[0].legend(
        handles=[
            Line2D([], [], marker="x", color="red", markersize=11, markeredgewidth=3, linestyle="None", label="Target"),
            Line2D([], [], color="#8b008b", linewidth=1.4, label="Robot Path (Spiral)"),
            Line2D(
                [],
                [],
                marker="o",
                color="#08519c",
                markerfacecolor="#08519c",
                markersize=9,
                linestyle="None",
                label="Attractive Pheromone",
            ),
        ],
        loc="upper right",
        frameon=True,
        facecolor="white",
        framealpha=0.95,
    )

    # axes[1].set_title("FOLLOW Behavior")
    axes[1].set_xlabel("W")
    axes[1].set_ylabel("H")
    axes[1].plot(
        gradient_path[:, 0],
        gradient_path[:, 1],
        color="black",
        linestyle="--",
        linewidth=1.8,
        zorder=4,
    )
    draw_gradient_arrows(axes[1], gradient_path)
    axes[1].plot(
        FOLLOWER_START[0],
        FOLLOWER_START[1],
        marker="o",
        color="green",
        markersize=8,
        linestyle="None",
        zorder=8,
    )
    axes[1].plot(
        TARGET[0],
        TARGET[1],
        marker="x",
        color="red",
        markersize=11,
        markeredgewidth=3,
        linestyle="None",
        zorder=8,
    )
    axes[1].legend(
        handles=[
            Line2D([], [], marker="o", color="green", markersize=8, linestyle="None", label="Follower Start"),
            FancyArrowPatch(
                (0, 0),
                (1, 0),
                arrowstyle="-|>",
                color="black",
                linestyle="--",
                linewidth=2,
                label="Gradient Ascent Direction",
            ),
        ],
        loc="upper right",
        frameon=True,
        facecolor="white",
        framealpha=0.95,
        handler_map={FancyArrowPatch: HandlerArrow()},
    )

    output_path = Path(output_path)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the convergence spiral and gradient-ascent pheromone figure."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"PNG file to write. Defaults to {OUTPUT_PATH}.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_convergence_spiral_gradient(args.output)
