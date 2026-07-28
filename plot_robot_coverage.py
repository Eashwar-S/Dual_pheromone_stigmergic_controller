# -*- coding: utf-8 -*-
"""Animate robot motion, swept coverage, and redundant cell coverage.

The input may be a metadata file, a run directory, or one of the ``robot_N``
frame directories inside a run. The final animation frame is always saved.
"""

import argparse
from datetime import datetime, timedelta
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Circle
import numpy as np


AREA_X_METERS = 4.0
AREA_Y_METERS = 1.8
ROBOT_RADIUS_METERS = 8.0 * 0.0254
DEFAULT_CELL_SIZE_METERS = 0.05
DEFAULT_INTERVAL_MS = 40


def resolve_metadata_path(input_path):
    path = Path(input_path).expanduser()
    if path.is_file():
        return path.resolve()
    if not path.is_dir():
        raise FileNotFoundError("Input path not found: {0}".format(path))

    direct_metadata = path / "metadata.json"
    if direct_metadata.is_file():
        return direct_metadata.resolve()

    # Camera frames live in run/robot_1, run/robot_2, or run/robot_3 while
    # metadata.json lives one level above them.
    parent_metadata = path.parent / "metadata.json"
    if path.name.lower().startswith("robot_") and parent_metadata.is_file():
        return parent_metadata.resolve()

    metadata_paths = sorted(path.rglob("metadata.json"))
    if len(metadata_paths) == 1:
        return metadata_paths[0].resolve()
    if metadata_paths:
        candidates = "\n  ".join(str(candidate) for candidate in metadata_paths)
        raise ValueError(
            "Multiple runs were found below {0}. Select a run or robot_N "
            "directory:\n  {1}".format(path, candidates)
        )
    raise FileNotFoundError(
        "No metadata.json was found in or below: {0}".format(path)
    )


def load_samples(metadata_path):
    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    samples = metadata.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("metadata.json does not contain a non-empty samples list")

    robot_names = sorted(
        {
            name
            for sample in samples
            for name in sample.get("robots", {})
            if "position" in sample.get("robots", {}).get(name, {})
        }
    )
    if not robot_names:
        raise ValueError("No robot positions were found in metadata.json")
    return metadata, samples, robot_names


def parse_timestamp(timestamp):
    """Parse an ISO timestamp, returning None when it is unavailable."""
    if not isinstance(timestamp, str) or not timestamp:
        return None
    if timestamp.endswith("Z"):
        timestamp = timestamp[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(timestamp)
    except ValueError:
        return None


def position_from_mapping(position):
    """Return an (x, y) tuple from a metadata position mapping."""
    if not isinstance(position, dict):
        return None
    try:
        return float(position["x"]), float(position["y"])
    except (KeyError, TypeError, ValueError):
        return None


def resolve_failure_state(metadata, samples):
    """Locate the failed robot and its fixed physical position in an E2 run."""
    failure_event = metadata.get("failure_event")
    if not isinstance(failure_event, dict):
        return None

    robot_name = failure_event.get("robot")
    if not isinstance(robot_name, str) or not robot_name:
        return None

    failure_time = parse_timestamp(failure_event.get("timestamp_utc"))
    if failure_time is None:
        start_time = parse_timestamp(samples[0].get("timestamp_utc"))
        try:
            elapsed = float(failure_event["elapsed_seconds"])
        except (KeyError, TypeError, ValueError):
            elapsed = None
        if start_time is not None and elapsed is not None:
            failure_time = start_time + timedelta(seconds=elapsed)

    failure_index = None
    if failure_time is not None:
        for index, sample in enumerate(samples):
            sample_time = parse_timestamp(sample.get("timestamp_utc"))
            if sample_time is not None and sample_time >= failure_time:
                failure_index = index
                break

    if failure_index is None:
        for index, sample in enumerate(samples):
            robot_data = sample.get("robots", {}).get(robot_name, {})
            status = str(robot_data.get("status", "")).lower()
            if status.startswith("failed"):
                failure_index = index
                break

    if failure_index is None:
        return None

    fixed_position = position_from_mapping(failure_event.get("position"))
    if fixed_position is None:
        after_position = robot_position(samples[failure_index], robot_name)
        before_index = max(0, failure_index - 1)
        before_position = robot_position(samples[before_index], robot_name)
        fixed_position = after_position or before_position

        # The failure timestamp usually falls between two samples. Interpolate
        # the position at that exact time instead of retaining later odometry
        # drift from a robot that has physically stopped.
        before_time = parse_timestamp(
            samples[before_index].get("timestamp_utc")
        )
        after_time = parse_timestamp(
            samples[failure_index].get("timestamp_utc")
        )
        if (
            failure_time is not None
            and before_position is not None
            and after_position is not None
            and before_time is not None
            and after_time is not None
            and after_time > before_time
        ):
            fraction = (
                (failure_time - before_time).total_seconds()
                / (after_time - before_time).total_seconds()
            )
            fraction = min(1.0, max(0.0, fraction))
            fixed_position = (
                before_position[0]
                + fraction * (after_position[0] - before_position[0]),
                before_position[1]
                + fraction * (after_position[1] - before_position[1]),
            )

    if fixed_position is None:
        return None

    return {
        "robot": robot_name,
        "sample_index": failure_index,
        "position": fixed_position,
    }


def effective_robot_position(sample, robot_name, sample_index, failure_state):
    """Return the displayed position, holding a failed robot stationary."""
    if (
        failure_state is not None
        and robot_name == failure_state["robot"]
        and sample_index >= failure_state["sample_index"]
    ):
        return failure_state["position"]
    return robot_position(sample, robot_name)


def contributes_coverage(robot_name, sample_index, failure_state):
    """Exclude a failed robot from coverage counts after its failure sample."""
    return not (
        failure_state is not None
        and robot_name == failure_state["robot"]
        and sample_index > failure_state["sample_index"]
    )


class CoverageGrid:
    """Track how many swept robot footprints cover each arena cell."""

    def __init__(self, cell_size):
        self.cell_size = float(cell_size)
        self.columns = int(math.ceil(AREA_X_METERS / self.cell_size))
        self.rows = int(math.ceil(AREA_Y_METERS / self.cell_size))
        self.counts = np.zeros((self.rows, self.columns), dtype=np.uint32)
        self.x_centers = (
            np.arange(self.columns, dtype=float) + 0.5
        ) * self.cell_size
        self.y_centers = (
            np.arange(self.rows, dtype=float) + 0.5
        ) * self.cell_size

    def cells_intersecting_circle(self, x, y):
        """Return cells whose rectangular area intersects the robot circle."""
        first_col = max(
            0, int(math.floor((x - ROBOT_RADIUS_METERS) / self.cell_size))
        )
        last_col = min(
            self.columns - 1,
            int(math.floor((x + ROBOT_RADIUS_METERS) / self.cell_size)),
        )
        first_row = max(
            0, int(math.floor((y - ROBOT_RADIUS_METERS) / self.cell_size))
        )
        last_row = min(
            self.rows - 1,
            int(math.floor((y + ROBOT_RADIUS_METERS) / self.cell_size)),
        )
        if first_col > last_col or first_row > last_row:
            return set()

        covered = set()
        radius_squared = ROBOT_RADIUS_METERS ** 2
        for row in range(first_row, last_row + 1):
            cell_y_min = row * self.cell_size
            cell_y_max = min((row + 1) * self.cell_size, AREA_Y_METERS)
            nearest_y = min(max(y, cell_y_min), cell_y_max)
            for col in range(first_col, last_col + 1):
                cell_x_min = col * self.cell_size
                cell_x_max = min(
                    (col + 1) * self.cell_size, AREA_X_METERS
                )
                nearest_x = min(max(x, cell_x_min), cell_x_max)
                if (
                    (nearest_x - x) ** 2 + (nearest_y - y) ** 2
                    <= radius_squared
                ):
                    covered.add((row, col))
        return covered

    def add_swept_footprint(self, previous, current):
        """Increment every cell touched by the disk swept along one segment."""
        if previous is None:
            segment_cells = self.cells_intersecting_circle(*current)
        else:
            distance = math.hypot(
                current[0] - previous[0], current[1] - previous[1]
            )
            interpolation_step = max(self.cell_size * 0.5, 0.005)
            point_count = max(1, int(math.ceil(distance / interpolation_step)))
            segment_cells = set()
            for index in range(point_count + 1):
                fraction = index / point_count
                x = previous[0] + fraction * (current[0] - previous[0])
                y = previous[1] + fraction * (current[1] - previous[1])
                segment_cells.update(self.cells_intersecting_circle(x, y))

        for row, col in segment_cells:
            self.counts[row, col] += 1


def robot_position(sample, robot_name):
    robot_data = sample.get("robots", {}).get(robot_name)
    if not robot_data:
        return None
    position = robot_data.get("position", {})
    try:
        return float(position["x"]), float(position["y"])
    except (KeyError, TypeError, ValueError):
        return None


def configure_axis(axis, title):
    axis.set_title(title)
    axis.set_xlim(0.0, AREA_X_METERS)
    axis.set_ylim(0.0, AREA_Y_METERS)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x position (m)")
    axis.set_ylabel("y position (m)")
    axis.grid(color="0.85", linewidth=0.5)


def animate(
    metadata_path,
    output_path,
    cell_size,
    interval_ms,
    frame_step,
    live_animation,
):
    metadata, samples, robot_names = load_samples(metadata_path)
    failure_state = resolve_failure_state(metadata, samples)
    grid = CoverageGrid(cell_size)
    colors_by_robot = {
        name: plt.get_cmap("tab10")(index % 10)
        for index, name in enumerate(robot_names)
    }
    paths = {name: [[], []] for name in robot_names}
    previous_positions = {name: None for name in robot_names}

    figure, (movement_axis, redundancy_axis) = plt.subplots(
        1, 2, figsize=(15, 5.8), constrained_layout=True
    )
    configure_axis(movement_axis, "Robot movement and covered cells")
    configure_axis(redundancy_axis, "Repeated coverage heat map")

    coverage_map = movement_axis.imshow(
        np.zeros_like(grid.counts, dtype=float),
        origin="lower",
        extent=(0.0, AREA_X_METERS, 0.0, AREA_Y_METERS),
        interpolation="nearest",
        cmap=colors.ListedColormap(["white", "#8fd694"]),
        norm=colors.BoundaryNorm([-0.5, 0.5, 1.5], 2),
        alpha=0.75,
        zorder=0,
    )
    redundancy_map = redundancy_axis.imshow(
        np.ma.masked_all(grid.counts.shape, dtype=float),
        origin="lower",
        extent=(0.0, AREA_X_METERS, 0.0, AREA_Y_METERS),
        interpolation="nearest",
        cmap="inferno",
        vmin=1.0,
        vmax=2.0,
        zorder=0,
    )
    colorbar = figure.colorbar(
        redundancy_map,
        ax=redundancy_axis,
        label="Repeated visits (coverage count - 1)",
        fraction=0.046,
        pad=0.04,
    )

    path_lines = {}
    robot_circles = {}
    robot_labels = {}
    for name in robot_names:
        color = colors_by_robot[name]
        path_lines[name], = movement_axis.plot(
            [], [], color=color, linewidth=1.4, label=name, zorder=2
        )
        robot_circles[name] = Circle(
            (0.0, 0.0),
            ROBOT_RADIUS_METERS,
            facecolor=color,
            edgecolor="black",
            linewidth=1.0,
            alpha=0.55,
            visible=False,
            zorder=3,
        )
        movement_axis.add_patch(robot_circles[name])
        robot_labels[name] = movement_axis.text(
            0.0, 0.0, "", fontsize=8, visible=False, zorder=4
        )
    movement_axis.legend(loc="upper right")

    status_text = figure.suptitle("")
    processed_index = -1

    def update_frame(sample_index):
        nonlocal processed_index
        for index in range(processed_index + 1, sample_index + 1):
            sample = samples[index]
            for name in robot_names:
                current = effective_robot_position(
                    sample, name, index, failure_state
                )
                if current is None:
                    continue
                previous = previous_positions[name]
                if contributes_coverage(name, index, failure_state):
                    grid.add_swept_footprint(previous, current)
                previous_positions[name] = current
                paths[name][0].append(current[0])
                paths[name][1].append(current[1])
        processed_index = sample_index

        for name in robot_names:
            path_lines[name].set_data(paths[name][0], paths[name][1])
            current = previous_positions[name]
            if current is None:
                continue
            robot_circles[name].center = current
            robot_circles[name].set_visible(True)
            robot_labels[name].set_position(
                (current[0] + ROBOT_RADIUS_METERS * 0.55,
                 current[1] + ROBOT_RADIUS_METERS * 0.55)
            )
            robot_labels[name].set_text(
                "{0}\n({1:.3f}, {2:.3f}) m".format(
                    name, current[0], current[1]
                )
            )
            robot_labels[name].set_visible(True)

        coverage_map.set_data((grid.counts > 0).astype(float))
        redundant = grid.counts.astype(float) - 1.0
        redundant[redundant <= 0.0] = np.nan
        redundancy_map.set_data(np.ma.masked_invalid(redundant))
        maximum_redundancy = max(1.0, float(np.nanmax(
            redundant
        )) if np.any(np.isfinite(redundant)) else 1.0)
        redundancy_map.set_clim(1.0, maximum_redundancy)
        colorbar.update_normal(redundancy_map)

        covered_cells = int(np.count_nonzero(grid.counts))
        redundant_cells = int(np.count_nonzero(grid.counts > 1))
        covered_percent = 100.0 * covered_cells / grid.counts.size
        timestamp = samples[sample_index].get("timestamp_utc", "unknown time")
        failure_text = ""
        if (
            failure_state is not None
            and sample_index >= failure_state["sample_index"]
        ):
            failure_text = (
                "  |  {0} held at failure position; later visits excluded"
            ).format(failure_state["robot"])
        status_text.set_text(
            "Sample {0}/{1}  |  {2}  |  covered: {3:.1f}%  |  "
            "redundant cells: {4}  |  radius: {5:.4f} m{6}".format(
                sample_index + 1,
                len(samples),
                timestamp,
                covered_percent,
                redundant_cells,
                ROBOT_RADIUS_METERS,
                failure_text,
            )
        )
        return (
            coverage_map,
            redundancy_map,
            status_text,
            *path_lines.values(),
            *robot_circles.values(),
            *robot_labels.values(),
        )

    frame_indices = list(range(0, len(samples), frame_step))
    if frame_indices[-1] != len(samples) - 1:
        frame_indices.append(len(samples) - 1)

    if live_animation:
        plt.ion()
    try:
        if live_animation:
            for sample_index in frame_indices:
                update_frame(sample_index)
                figure.canvas.draw_idle()
                plt.pause(interval_ms / 1000.0)
            plt.ioff()
            figure.savefig(output_path, dpi=180, bbox_inches="tight")
            print("Saved final coverage image: {0}".format(output_path))
            plt.show()
        else:
            update_frame(len(samples) - 1)
            figure.canvas.draw_idle()
            plt.show()
            figure.savefig(output_path, dpi=180, bbox_inches="tight")
            print(
                "Closed final coverage window and saved image: {0}".format(
                    output_path
                )
            )
    finally:
        plt.close(figure)

    coverage_status = metadata.get("coverage", {})
    if coverage_status:
        print("Metadata coverage status: {0}".format(coverage_status))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Animate RoboMaster trajectories and grid coverage from metadata.json."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help=(
            "metadata.json, its run folder, or a robot_N folder below the run"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Final PNG path (default: <run folder>/coverage_final.png)",
    )
    parser.add_argument(
        "--cell-size",
        type=float,
        default=DEFAULT_CELL_SIZE_METERS,
        help="Coverage grid cell size in meters (default: %(default)s)",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=DEFAULT_INTERVAL_MS,
        help="Delay between displayed frames (default: %(default)s)",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Display every Nth metadata sample while still processing all samples",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help=(
            "Skip the live animation, display only the completed coverage "
            "figure, and save it after the window is closed"
        ),
    )
    args = parser.parse_args()
    if args.cell_size <= 0.0:
        parser.error("--cell-size must be greater than zero")
    if args.interval_ms < 0:
        parser.error("--interval-ms cannot be negative")
    if args.frame_step <= 0:
        parser.error("--frame-step must be greater than zero")
    return args


def main():
    args = parse_args()
    metadata_path = resolve_metadata_path(args.input)
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else metadata_path.parent / "coverage_final.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animate(
        metadata_path=metadata_path,
        output_path=output_path,
        cell_size=args.cell_size,
        interval_ms=args.interval_ms,
        frame_step=args.frame_step,
        live_animation=not args.no_show,
    )


if __name__ == "__main__":
    main()
