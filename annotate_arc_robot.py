from pathlib import Path
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Rectangle
from PIL import Image, ImageEnhance


def polar_point(center, radius, angle_degrees):
    angle = math.radians(angle_degrees)
    return (
        center[0] + radius * math.cos(angle),
        center[1] + radius * math.sin(angle),
    )


def annotate_arc_robot(
    input_path="arc_robot.png",
    output_path="arc_robot_annotated.png",
):
    image_path = Path(input_path)
    output_path = Path(output_path)

    image = Image.open(image_path).convert("RGB")
    image = ImageEnhance.Contrast(image).enhance(1.16)
    image = ImageEnhance.Brightness(image).enhance(1.03)

    width, height = image.size
    robot_center = (width * 0.514, height * 0.545)
    cell = width * 0.25
    fov_origin = (robot_center[0] + 0.10 * cell, robot_center[1] - 0.42 * cell)
    fov_radius = cell * 1.02
    collision_radius = fov_radius

    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=300)
    ax.imshow(image)
    ax.axis("off")

    grid_color = "#222222"
    x0 = robot_center[0] - 1.5 * cell
    y0 = robot_center[1] - 1.5 * cell
    for index in range(4):
        x = x0 + index * cell
        y = y0 + index * cell
        ax.plot([x, x], [y0, y0 + 3 * cell], color=grid_color, linewidth=1.25, alpha=0.78, zorder=2)
        ax.plot([x0, x0 + 3 * cell], [y, y], color=grid_color, linewidth=1.25, alpha=0.78, zorder=2)

    sectors = [
        ("left", -150, -110, "#9ecae1"),
        ("straight", -110, -70, "#c7e9c0"),
        ("right", -70, -30, "#fdd0a2"),
    ]
    for _, theta1, theta2, color in sectors:
        ax.add_patch(
            Wedge(
                fov_origin,
                fov_radius,
                theta1,
                theta2,
                facecolor=color,
                edgecolor="none",
                alpha=0.42,
                zorder=3,
            )
        )

    for angle in [-150, -110, -70, -30]:
        end = polar_point(fov_origin, fov_radius, angle)
        ax.plot([fov_origin[0], end[0]], [fov_origin[1], end[1]], color="black", linewidth=1.7, zorder=5)

    ax.add_patch(
        Wedge(
            fov_origin,
            fov_radius,
            -150,
            -30,
            facecolor="none",
            edgecolor="black",
            linewidth=2.0,
            zorder=5,
        )
    )
    ax.add_patch(
        Wedge(
            fov_origin,
            collision_radius,
            -150,
            -30,
            facecolor="none",
            edgecolor="#cc0000",
            linewidth=2.2,
            alpha=0.9,
            zorder=6,
        )
    )

    sector_label_positions = {
        "left": polar_point(fov_origin, fov_radius * 0.68, -132),
        "straight": polar_point(fov_origin, fov_radius * 0.70, -90),
        "right": polar_point(fov_origin, fov_radius * 0.68, -48),
    }
    for label, position in sector_label_positions.items():
        ax.text(
            position[0],
            position[1],
            label,
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=7,
        )

    legend_x = x0 + 1.92 * cell
    legend_y = y0 + 2.24 * cell
    ax.add_patch(
        Rectangle(
            (legend_x - 10, legend_y - 10),
            172,
            108,
            facecolor="white",
            edgecolor="none",
            alpha=1.0,
            zorder=8,
        )
    )
    ax.add_patch(
        Wedge(
            (legend_x + 18, legend_y + 24),
            22,
            -150,
            -30,
            facecolor="none",
            edgecolor="black",
            linewidth=1.0,
            zorder=9,
        )
    )
    ax.text(legend_x + 50, legend_y + 24, "120 deg FOV", fontsize=9, va="center", zorder=9)
    for theta1, theta2, color in [(-150, -110, "#9ecae1"), (-110, -70, "#c7e9c0"), (-70, -30, "#fdd0a2")]:
        ax.add_patch(
            Wedge(
                (legend_x + 18, legend_y + 57),
                22,
                theta1,
                theta2,
                facecolor=color,
                edgecolor="black",
                linewidth=0.6,
                alpha=0.75,
                zorder=9,
            )
        )
    ax.text(legend_x + 50, legend_y + 56, "3 actions", fontsize=9, va="center", zorder=9)
    ax.add_patch(
        Wedge(
            (legend_x + 18, legend_y + 82),
            18,
            -150,
            -30,
            facecolor="none",
            edgecolor="#cc0000",
            linewidth=1.6,
            zorder=9,
        )
    )
    ax.text(legend_x + 50, legend_y + 82, "collision FOV", fontsize=9, va="center", zorder=9)

    ax.text(
        x0 + 1.5 * cell,
        y0 + 3.10 * cell,
        "W",
        fontsize=16,
        ha="center",
        va="top",
        zorder=9,
    )
    ax.text(
        x0 - 0.16 * cell,
        y0 + 1.5 * cell,
        "H",
        fontsize=16,
        ha="center",
        va="center",
        rotation=90,
        zorder=9,
    )

    ax.set_xlim(x0 - 0.36 * cell, x0 + 3.34 * cell)
    ax.set_ylim(y0 + 3.34 * cell, y0 - 0.22 * cell)

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0, facecolor="white")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    annotate_arc_robot()
