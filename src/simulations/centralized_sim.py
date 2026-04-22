import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Set, Tuple
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common.simulation import FrameWriter, compute_fps, make_writer, run_animation
from src.common.utilities import generate_unique_targets, mark_visible, discover_targets_in_vnhood
from src.common.geometry import manhattan_path
from src.common.visualization import (coverage_to_image, region_colors, 
                                                   draw_voronoi_borders, save_targets_over_time_plot)
from src.centralized.partitioning import lloyd_balanced
from src.centralized.path_planning import sensor_aware_path_for_region
from src.centralized.robot import Robot


def maybe_trigger_failure(global_step, fail_at_step, fail_robot_id, robots, failure_triggered):
    """Trigger failure at specified step."""
    if failure_triggered or fail_robot_id is None or fail_at_step is None:
        return failure_triggered
    if global_step == fail_at_step:
        robots[fail_robot_id].failed = True
        return True
    return failure_triggered


def first_finisher_after_failure(robots, fail_robot_id, failure_triggered):
    """Return first non-failed robot that finished its path."""
    if not failure_triggered:
        return None
    for r in robots:
        if r.id == fail_robot_id:
            continue
        if not r.failed and r.idx >= len(r.path) - 1:
            return r
    return None


def extend_full_pts_for_robot(robot_id, extra_path, remaining_scatters):
    """Update path visualization cache."""
    rem_sc, full_pts = remaining_scatters[robot_id]
    if len(extra_path):
        add_pts = np.array([[x + 0.5, y + 0.5] for (x, y) in extra_path])
        full_pts = np.vstack([full_pts, add_pts])
        remaining_scatters[robot_id] = (rem_sc, full_pts)


def maybe_reallocate_failed_path(robots, fail_robot_id, failure_triggered, failure_reallocated, remaining_scatters):
    """Reallocate failed robot's path to first finisher."""
    if not failure_triggered or failure_reallocated:
        return failure_reallocated
    
    takeover = first_finisher_after_failure(robots, fail_robot_id, failure_triggered)
    if takeover is None:
        return failure_reallocated
    
    failed = robots[fail_robot_id]
    nav = manhattan_path(takeover.pos, failed.pos)
    rem = failed.path[failed.idx:]
    
    extension = nav[1:]
    if len(rem) > 1:
        extension += rem[1:]
    
    takeover.path.extend(extension)
    extend_full_pts_for_robot(takeover.id, extension, remaining_scatters)
    
    failed.path = failed.path[:failed.idx + 1]
    extend_full_pts_for_robot(failed.id, [], remaining_scatters)
    
    return True


def sim_step(robots, covered, targets, found_targets, W, H, global_step, fail_at_step, 
             fail_robot_id, failure_triggered, failure_reallocated, targets_found_over_time, 
             remaining_scatters):
    """Execute one simulation step."""
    failure_triggered = maybe_trigger_failure(global_step, fail_at_step, fail_robot_id, robots, failure_triggered)
    
    for r in robots:
        x, y = r.pos
        mark_visible(covered, x, y, r=ROBOT_RADIUS)
        discover_targets_in_vnhood(x, y, targets, found_targets, W, H, r=ROBOT_RADIUS)
    
    for r in robots:
        r.step(robots)
    
    for r in robots:
        x, y = r.pos
        mark_visible(covered, x, y, r=ROBOT_RADIUS)
        discover_targets_in_vnhood(x, y, targets, found_targets, W, H, r=ROBOT_RADIUS)
    
    failure_reallocated = maybe_reallocate_failed_path(robots, fail_robot_id, failure_triggered, 
                                                        failure_reallocated, remaining_scatters)
    
    global_step += 1
    targets_found_over_time.append(len(found_targets))
    
    if len(found_targets) >= len(targets):
        print(f"\nAll targets discovered at step {global_step}")
        sys.exit(0)
    
    return global_step, failure_triggered, failure_reallocated


def update(frame, robots, covered, targets, found_targets, W, H, global_step, fail_at_step,
           fail_robot_id, failure_triggered, failure_reallocated, targets_found_over_time,
           remaining_scatters, visited_scatters, world_img, shared_img, robot_scatter,
           robot_labels, disc_plot, und_plot, ax_world, fig, frame_writer, steps_per_frame,
           output_dir, plot_saved, state_dict):
    """Animation update function."""
    for _ in range(steps_per_frame):
        global_step, failure_triggered, failure_reallocated = sim_step(
            robots, covered, targets, found_targets, W, H, global_step, fail_at_step,
            fail_robot_id, failure_triggered, failure_reallocated, targets_found_over_time, 
            remaining_scatters
        )
    
    state_dict['global_step'] = global_step
    state_dict['failure_triggered'] = failure_triggered
    state_dict['failure_reallocated'] = failure_reallocated
    
    img = coverage_to_image(covered)
    world_img.set_data(img)
    shared_img.set_data(img)
    
    positions = np.array([[r.pos[0] + 0.5, r.pos[1] + 0.5] for r in robots])
    robot_scatter.set_offsets(positions)
    
    colors = ['red' if r.failed else 'k' for r in robots]
    robot_scatter.set_facecolors(colors)
    robot_scatter.set_edgecolors(colors)
    
    for i, r in enumerate(robots):
        robot_labels[i].set_position((r.pos[0] + 0.6, r.pos[1] + 0.6))
        robot_labels[i].set_text(f"R{r.id}" + (" (failed)" if r.failed else ""))
        robot_labels[i].set_color('red' if r.failed else 'k')
    
    for i, r in enumerate(robots):
        rem_sc, full_pts = remaining_scatters[i]
        vis_sc, _ = visited_scatters[i]
        v_pts = full_pts[:r.idx + 1]
        rem_pts = full_pts[r.idx + 1:]
        vis_sc.set_offsets(v_pts if len(v_pts) else np.empty((0, 2)))
        rem_sc.set_offsets(rem_pts if len(rem_pts) else np.empty((0, 2)))
    
    disc = list(found_targets)
    und = list(targets - found_targets)
    if disc:
        dx, dy = zip(*disc)
        disc_plot.set_offsets(np.c_[[x + 0.5 for x in dx], [y + 0.5 for y in dy]])
    else:
        disc_plot.set_offsets(np.empty((0, 2)))
    if und:
        ux, uy = zip(*und)
        und_plot.set_offsets(np.c_[[x + 0.5 for x in ux], [y + 0.5 for y in uy]])
    else:
        und_plot.set_offsets(np.empty((0, 2)))
    
    ax_world.set_title(
        "World — Equal-Area Voronoi Partition (Centralized)\n"
        f"Covered: {covered.sum()} / {W*H} cells, Found targets: {len(found_targets)} / {len(targets)}"
    )
    
    if (not plot_saved) and all(r.idx >= len(r.path) - 1 for r in robots):
        output_dir.mkdir(parents=True, exist_ok=True)
        save_targets_over_time_plot(output_dir / "targets_over_time.png", targets_found_over_time)
        
        if fail_robot_id is not None:
            np.save(f'{output_dir}/centralized_approach_with_failure.npy', np.array(targets_found_over_time))
        else:
            np.save(f'{output_dir}/centralized_approach_without_failure.npy', np.array(targets_found_over_time))
        plot_saved = True
        state_dict['plot_saved'] = plot_saved
    
    frame_writer.save(fig)
    return (world_img, shared_img, robot_scatter, und_plot, disc_plot, *robot_labels,
            *[sc for sc, _ in remaining_scatters], *[sc for sc, _ in visited_scatters])


if __name__ == "__main__":
    GRID_SIZE = 100
    N_ROBOTS = 10
    N_TARGETS = 5
    STEPS_PER_FRAME = 10
    INTERVAL_MS = 50
    RANDOM_SEED = 7
    ROBOT_RADIUS = 3
    
    targets_found_over_time = []
    plot_saved = False
    OUTPUT_DIR = Path('output_frames/centralized_approach')
    
    FAIL_ROBOT_ID = 1
    FAIL_AT_STEP = 20
    global_step = 0
    failure_triggered = False
    failure_reallocated = False
    
    # partitioning parameters
    MAX_ITERS_ASSIGN = 30
    MAX_ITERS_CENTERS = 10
    LAMBDA_STEP0 = 0.1
    LAMBDA_DECAY = 0.1
    
    rng = np.random.default_rng(RANDOM_SEED)
    FPS = compute_fps(INTERVAL_MS)
    writer = make_writer(INTERVAL_MS, title="Centralized Swarm", artist="you")
    frame_writer = FrameWriter("output_frames/centralized_approach/")
    
    W = H = GRID_SIZE
    yy, xx = np.mgrid[0:H, 0:W]
    points = np.column_stack((xx.ravel() + 0.5, yy.ravel() + 0.5))
    
    labels, centers = lloyd_balanced(points, N_ROBOTS, MAX_ITERS_CENTERS, MAX_ITERS_ASSIGN,
                                     LAMBDA_STEP0, LAMBDA_DECAY, rng)
    zones = labels.reshape(H, W)
    
    masks = [(zones == i) for i in range(N_ROBOTS)]
    sweeping_paths = []
    for i in range(N_ROBOTS):
        p = sensor_aware_path_for_region(masks[i], robot_radius=ROBOT_RADIUS)
        if not p:
            cx, cy = centers[i]
            x = int(np.clip(round(cx - 0.5), 0, W - 1))
            y = int(np.clip(round(cy - 0.5), 0, H - 1))
            p = [(x, y)]
        sweeping_paths.append(p)
    
    # Generate random starting positions for robots
    start_positions = []
    for i in range(N_ROBOTS):
        x = rng.integers(0, W)
        y = rng.integers(0, H)
        start_positions.append((x, y))
    
    # Build full paths: random start -> navigate to sweep path start -> sweep path
    full_paths = []
    for i in range(N_ROBOTS):
        sweeping_path = sweeping_paths[i]
        start_pos = start_positions[i]
        if sweeping_path:
            nav_path = manhattan_path(start_pos, sweeping_path[0])
            full_path = nav_path[:-1] + sweeping_path
        else:
            full_path = [start_pos]
        full_paths.append(full_path)
    
    robots = [Robot(i, full_paths[i]) for i in range(N_ROBOTS)]
    covered = np.zeros((H, W), dtype=bool)
    targets = generate_unique_targets(GRID_SIZE, N_TARGETS, rng)
    found_targets: Set[Tuple[int, int]] = set()
    
    fig = plt.figure(figsize=(12.0, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.12)
    ax_world = fig.add_subplot(gs[0, 0])
    ax_shared = fig.add_subplot(gs[0, 1])
    
    zone_rgba, _ = region_colors(zones, alpha=0.12)
    ax_world.imshow(zone_rgba, origin='lower', extent=[0, W, 0, H])
    world_img = ax_world.imshow(coverage_to_image(covered), origin='lower', extent=[0, W, 0, H], vmin=0.0, vmax=1.0)
    
    draw_voronoi_borders(ax_world, zones, color='#003366', lw=1.2, alpha=0.9)
    
    ax_world.set_title("World — Equal-Area Voronoi Partition (Centralized)")
    ax_world.set_xlim(0, W)
    ax_world.set_ylim(0, H)
    ax_world.set_aspect('equal', adjustable='box')
    ax_world.set_xticks(np.arange(0, W + 1, 10))
    ax_world.set_yticks(np.arange(0, H + 1, 10))
    ax_world.set_xticks(np.arange(0, W + 1, 1), minor=True)
    ax_world.set_yticks(np.arange(0, H + 1, 1), minor=True)
    ax_world.grid(which='major', color='k', alpha=0.15, linewidth=0.5)
    ax_world.grid(which='minor', color='k', alpha=0.05, linewidth=0.2)
    
    robot_colors = ['k' for _ in robots]
    robot_scatter = ax_world.scatter([r.pos[0] + 0.5 for r in robots],
                                     [r.pos[1] + 0.5 for r in robots],
                                     s=40, marker='o', c=robot_colors, zorder=4)
    
    robot_labels = []
    for r in robots:
        txt = ax_world.text(r.pos[0] + 0.6, r.pos[1] + 0.6,
                           f"R{r.id}", fontsize=7, color='k', zorder=6)
        robot_labels.append(txt)
    
    remaining_scatters, visited_scatters = [], []
    for r in robots:
        full_pts = np.array([[x + 0.5, y + 0.5] for (x, y) in r.path])
        rem_sc = ax_world.scatter(full_pts[:, 0], full_pts[:, 1], s=6, marker='s',
                                 facecolors='none', edgecolors='0.35', alpha=0.14, linewidths=0.45, zorder=1)
        vis_sc = ax_world.scatter([], [], s=10, marker='s',
                                 facecolors='none', edgecolors='0.2', alpha=0.55, linewidths=0.6, zorder=2)
        remaining_scatters.append((rem_sc, full_pts))
        visited_scatters.append((vis_sc, []))
    
    if targets:
        tx_world, ty_world = zip(*targets)
    else:
        tx_world, ty_world = [], []
    ax_world.scatter([x + 0.5 for x in tx_world], [y + 0.5 for y in ty_world],
                    s=20, marker='x', c='r', alpha=0.9, zorder=4)
    
    ax_shared.imshow(zone_rgba, origin='lower', extent=[0, W, 0, H])
    shared_img = ax_shared.imshow(coverage_to_image(covered), origin='lower', extent=[0, W, 0, H], vmin=0.0, vmax=1.0)
    
    ax_shared.set_title("Shared Map (Global Knowledge)")
    ax_shared.set_xlim(0, W)
    ax_shared.set_ylim(0, H)
    ax_shared.set_aspect('equal', adjustable='box')
    ax_shared.set_xticks(np.arange(0, W + 1, 10))
    ax_shared.set_yticks(np.arange(0, H + 1, 10))
    ax_shared.set_xticks(np.arange(0, W + 1, 1), minor=True)
    ax_shared.set_yticks(np.arange(0, H + 1, 1), minor=True)
    ax_shared.grid(which='major', color='k', alpha=0.15, linewidth=0.5)
    ax_shared.grid(which='minor', color='k', alpha=0.05, linewidth=0.2)
    
    disc_plot = ax_shared.scatter([], [], s=25, marker='o',
                                 facecolors='none', edgecolors='g', linewidths=1.5, label='Discovered')
    und_tx, und_ty = (list(zip(*targets)) if targets else ([], []))
    und_plot = ax_shared.scatter([x + 0.5 for x in und_tx], [y + 0.5 for y in und_ty],
                                s=18, marker='x', c='r', label='Undiscovered')
    ax_shared.legend(loc='upper right', fontsize=8, frameon=False)
    
    state_dict = {
        'global_step': global_step,
        'failure_triggered': failure_triggered,
        'failure_reallocated': failure_reallocated,
        'plot_saved': plot_saved
    }
    
    def update_wrapper(frame):
        return update(frame, robots, covered, targets, found_targets, W, H, 
                     state_dict['global_step'], FAIL_AT_STEP,
                     FAIL_ROBOT_ID, state_dict['failure_triggered'], state_dict['failure_reallocated'], 
                     targets_found_over_time, remaining_scatters, visited_scatters, world_img, shared_img, 
                     robot_scatter, robot_labels, disc_plot, und_plot, ax_world, fig, frame_writer, 
                     STEPS_PER_FRAME, OUTPUT_DIR, state_dict['plot_saved'], state_dict)
    
    anim = run_animation(fig, update_wrapper, frames=2000000, interval_ms=INTERVAL_MS, blit=False)
    plt.show()
