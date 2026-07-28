from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from PIL import Image, ImageEnhance


def annotate_robotarium_robot(
    input_path="robotarium_robot.png",
    output_path="robotarium_robot_annotated.png",
):
    image_path = Path(input_path)
    output_path = Path(output_path)

    image = Image.open(image_path).convert("RGB")
    image = ImageEnhance.Contrast(image).enhance(1.22)
    image = ImageEnhance.Brightness(image).enhance(1.04)

    width, height = image.size
    robot_center = (width * 0.499, height * 0.514)
    cell = width * 0.12
    robot_radius = cell * 0.44
    collision_radius_cells = 1.5

    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=300)
    ax.imshow(image)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")

    grid_color = "#222222"
    x0 = robot_center[0] - 2.5 * cell
    y0 = robot_center[1] - 2.5 * cell
    for i in range(6):
        x = x0 + i * cell
        y = y0 + i * cell
        ax.plot([x, x], [y0, y0 + 5 * cell], color=grid_color, linewidth=1.25, alpha=0.78, zorder=2)
        ax.plot([x0, x0 + 5 * cell], [y, y], color=grid_color, linewidth=1.25, alpha=0.78, zorder=2)

    radius_cells = {
        "N": (robot_center[0] - cell / 2, robot_center[1] - 1.5 * cell),
        "W": (robot_center[0] - 1.5 * cell, robot_center[1] - cell / 2),
        "E": (robot_center[0] + cell / 2, robot_center[1] - cell / 2),
        "S": (robot_center[0] - cell / 2, robot_center[1] + cell / 2),
    }
    for x, y in radius_cells.values():
        ax.add_patch(
            Rectangle(
                (x, y),
                cell,
                cell,
                facecolor="#bfbfbf",
                edgecolor="black",
                linewidth=1.0,
                alpha=0.58,
                zorder=3,
            )
        )

    ax.add_patch(
        Circle(
            robot_center,
            collision_radius_cells * cell,
            facecolor="none",
            edgecolor="#cc0000",
            linewidth=2.2,
            alpha=0.9,
            zorder=5,
        )
    )

    action_style = dict(
        arrowstyle="-|>",
        color="black",
        linewidth=2.0,
        mutation_scale=14,
        zorder=6,
    )
    action_directions = {
        "N": (0, -1),
        "W": (-1, 0),
        "E": (1, 0),
        "S": (0, 1),
    }
    label_offsets = {
        "N": (0, -18, "center", "bottom"),
        "W": (-18, 0, "right", "center"),
        "E": (18, 0, "left", "center"),
        "S": (0, 18, "center", "top"),
    }
    for label, (dx, dy) in action_directions.items():
        start = (robot_center[0] + dx * robot_radius, robot_center[1] + dy * robot_radius)
        end = (robot_center[0] + dx * cell, robot_center[1] + dy * cell)
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                **action_style,
            )
        )
        offset_x, offset_y, ha, va = label_offsets[label]
        ax.text(
            end[0] + offset_x,
            end[1] + offset_y,
            label,
            fontsize=11,
            fontweight="bold",
            color="black",
            ha=ha,
            va=va,
            zorder=6,
        )

    legend_x = x0 + 2.85 * cell
    legend_y = y0 + 0.18 * cell
    ax.add_patch(
        Rectangle(
            (legend_x - 10, legend_y - 10),
            228,
            116,
            facecolor="white",
            edgecolor="none",
            alpha=0.94,
            zorder=7,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (legend_x, legend_y + 18),
            (legend_x + 48, legend_y + 18),
            arrowstyle="-|>",
            color="black",
            linewidth=1.3,
            mutation_scale=14,
            zorder=8,
        )
    )
    ax.text(legend_x + 62, legend_y + 18, "actions", fontsize=12, va="center", zorder=8)
    ax.add_patch(
        Rectangle(
            (legend_x, legend_y + 42),
            28,
            28,
            facecolor="#bfbfbf",
            edgecolor="black",
            linewidth=0.8,
            zorder=8,
        )
    )
    ax.text(legend_x + 62, legend_y + 56, "radius 1", fontsize=12, va="center", zorder=8)
    ax.add_patch(
        Circle(
            (legend_x + 14, legend_y + 92),
            14,
            facecolor="none",
            edgecolor="#cc0000",
            linewidth=1.6,
            zorder=8,
        )
    )
    ax.text(
        legend_x + 62,
        legend_y + 92,
        "collision r = 1",
        fontsize=12,
        va="center",
        zorder=8,
    )

    ax.text(
        x0 + 2.5 * cell,
        y0 + 5.14 * cell,
        "W",
        fontsize=16,
        ha="center",
        va="top",
        zorder=8,
    )
    ax.text(
        x0 - 0.28 * cell,
        y0 + 2.5 * cell,
        "H",
        fontsize=16,
        ha="center",
        va="center",
        rotation=90,
        zorder=8,
    )

    ax.set_xlim(x0 - 0.60 * cell, x0 + 5.40 * cell)
    ax.set_ylim(y0 + 5.42 * cell, y0 - 0.32 * cell)

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0, facecolor="white")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    annotate_robotarium_robot()
