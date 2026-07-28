from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


def draw_simulation_environment(output_path="simulation_environment.png"):
    """Draw a grid-world schematic for the swarm simulation environment."""
    grid_size = 5
    robot_cell = (2, 2)
    robot_center = (robot_cell[0] + 0.5, robot_cell[1] + 0.5)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    radius_cells = {
        "N": (robot_cell[0], robot_cell[1] + 1),
        "W": (robot_cell[0] - 1, robot_cell[1]),
        "E": (robot_cell[0] + 1, robot_cell[1]),
        "S": (robot_cell[0], robot_cell[1] - 1),
    }
    for x, y in radius_cells.values():
        ax.add_patch(
            Rectangle(
                (x, y),
                1,
                1,
                facecolor="#d9d9d9",
                edgecolor="none",
                zorder=1,
            )
        )

    ax.add_patch(
        Rectangle(
            robot_cell,
            1,
            1,
            facecolor="#f2f2f2",
            edgecolor="none",
            zorder=1,
        )
    )

    for coord in range(grid_size + 1):
        ax.plot([0, grid_size], [coord, coord], color="#8f8f8f", linewidth=1.35, alpha=1.0, zorder=2)
        ax.plot([coord, coord], [0, grid_size], color="#8f8f8f", linewidth=1.35, alpha=1.0, zorder=2)

    ax.add_patch(
        Circle(
            robot_center,
            0.34,
            facecolor="#222222",
            edgecolor="black",
            linewidth=1.3,
            zorder=4,
        )
    )

    arrow_length = 1.35
    arrow_style = dict(
        arrowstyle="-|>",
        color="black",
        linewidth=2.1,
        mutation_scale=16,
        shrinkA=7,
        shrinkB=0,
        zorder=6,
    )
    action_vectors = [(arrow_length, 0), (-arrow_length, 0), (0, arrow_length), (0, -arrow_length)]
    for dx, dy in action_vectors:
        ax.annotate(
            "",
            xy=(robot_center[0] + dx, robot_center[1] + dy),
            xytext=robot_center,
            arrowprops=arrow_style,
        )

    label_style = dict(fontsize=12, fontweight="bold", ha="left", va="top", zorder=7)
    for label, (x, y) in radius_cells.items():
        ax.text(x + 0.08, y + 0.92, label, **label_style)

    legend_x = grid_size - 1.45
    legend_top = grid_size - 0.12
    ax.add_patch(
        Rectangle(
            (legend_x - 0.10, legend_top - 1.08),
            1.42,
            1.08,
            facecolor="white",
            edgecolor="none",
            zorder=7,
        )
    )
    # ax.text(
    #     legend_x,
    #     legend_top,
    #     "Grid world: W = H",
    #     fontsize=12,
    #     ha="left",
    #     va="top",
    #     zorder=8,
    # )
    ax.add_patch(
        FancyArrowPatch(
            (legend_x, legend_top - 0.24),
            (legend_x + 0.32, legend_top - 0.24),
            arrowstyle="-|>",
            color="black",
            linewidth=1.1,
            mutation_scale=16,
            zorder=8,
        )
    )
    ax.text(legend_x + 0.42, legend_top - 0.24, "actions", fontsize=12, ha="left", va="center", zorder=8)
    ax.add_patch(
        Rectangle(
            (legend_x, legend_top - 0.62),
            0.24,
            0.24,
            facecolor="#d9d9d9",
            edgecolor="black",
            linewidth=0.8,
            zorder=8,
        )
    )
    ax.text(legend_x + 0.42, legend_top - 0.50, "radius 1", fontsize=12, ha="left", va="center", zorder=8)
    ax.add_patch(
        Circle(
            (legend_x + 0.12, legend_top - 0.84),
            0.12,
            facecolor="#222222",
            edgecolor="black",
            linewidth=0.8,
            zorder=8,
        )
    )
    ax.text(legend_x + 0.42, legend_top - 0.84, "robot", fontsize=12, ha="left", va="center", zorder=8)
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("W", fontsize=17, labelpad=10)
    ax.set_ylabel("H", fontsize=17, labelpad=10)
    ax.set_xticks(range(grid_size + 1))
    ax.set_yticks(range(grid_size + 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.6)

    output_path = Path(output_path)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    draw_simulation_environment()
