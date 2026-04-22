import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Set, Tuple
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common.simulation import FrameWriter, compute_fps, run_animation
from src.common.utilities import generate_unique_targets, mark_visible, discover_targets_in_vnhood
from src.common.visualization import coverage_to_image, save_targets_over_time_plot
from src.stigmergy.pheromone import apply_decay
from src.stigmergy.robot_efficient import Robot


def pheromone_to_rgba(p_rep: np.ndarray) -> np.ndarray:
    """Convert repulsive pheromone field to RGBA visualization."""
    H, W = p_rep.shape
    rgba = np.zeros((H, W, 4), dtype=float)
    max_rep = np.percentile(p_rep, 98) if np.max(p_rep) > 0 else 1.0
    norm_rep = np.clip(p_rep / (max_rep + 1e-9), 0, 1) * 0.5
    
    # Red-tinted pheromone visualization
    rgba[..., 0] = norm_rep * 1.0  # Red channel
    rgba[..., 1] = norm_rep * 0.2  # Green channel
    rgba[..., 2] = norm_rep * 0.6  # Blue channel
    rgba[..., 3] = np.clip(norm_rep, 0, 1)  # Alpha
    return np.clip(rgba, 0.0, 1.0)


def maybe_trigger_failure(global_step, fail_at_step, fail_robot_id, robots, failure_triggered):
    """Trigger failure at specified step."""
    if failure_triggered or fail_robot_id is None or fail_at_step is None:
        return failure_triggered
    if global_step == fail_at_step:
        robots[fail_robot_id].failed = True
        print(f"Robot {fail_robot_id} failed at step {global_step}")
        return True
    return failure_triggered


def sim_step(robots, pheromone, covered, targets, found_targets, W, H, global_step,
             fail_at_step, fail_robot_id, failure_triggered, targets_found_over_time,
             robot_radius, collision_radius, pher_deposit, tau_decay, pher_min):
    """Execute one simulation step."""
    # Apply pheromone decay
    apply_decay(pheromone, tau_decay, pher_min)
    
    # Handle failures
    failure_triggered = maybe_trigger_failure(global_step, fail_at_step, fail_robot_id, robots, failure_triggered)
    
    # Pre-move sensing
    for r in robots:
        if not r.failed:
            mark_visible(r.local_covered, r.x, r.y, robot_radius)
            mark_visible(covered, r.x, r.y, robot_radius)
            discover_targets_in_vnhood(r.x, r.y, targets, found_targets, W, H, robot_radius)
    
    # Execute moves
    for r in robots:
        if not r.failed:
            r.step(pheromone, robots, robot_radius, collision_radius)
            # Deposit pheromone after moving
            r.deposit_pheromone(pheromone, pher_deposit, robot_radius)
    
    # Post-move sensing
    for r in robots:
        if not r.failed:
            mark_visible(r.local_covered, r.x, r.y, robot_radius)
            mark_visible(covered, r.x, r.y, robot_radius)
            discover_targets_in_vnhood(r.x, r.y, targets, found_targets, W, H, robot_radius)
    
    global_step += 1
    targets_found_over_time.append(len(found_targets))
    
    # Check termination
    if len(found_targets) >= len(targets):
        print(f"\n✓ All targets discovered at step {global_step}")
        sys.exit(0)
    
    return global_step, failure_triggered


def update(frame, robots, pheromone, covered, targets, found_targets, W, H, global_step,
           fail_at_step, fail_robot_id, failure_triggered, targets_found_over_time,
           world_img, pher_img, shared_img, robot_scatter, robot_labels, disc_plot, und_plot,
           ax_world, fig, frame_writer, steps_per_frame, output_dir, plot_saved,
           robot_radius, collision_radius, pher_deposit, tau_decay, pher_min, state_dict):
    """Animation update function."""
    for _ in range(steps_per_frame):
        global_step, failure_triggered = sim_step(
            robots, pheromone, covered, targets, found_targets, W, H, global_step,
            fail_at_step, fail_robot_id, failure_triggered, targets_found_over_time,
            robot_radius, collision_radius, pher_deposit, tau_decay, pher_min
        )
    
    state_dict['global_step'] = global_step
    state_dict['failure_triggered'] = failure_triggered
    
    # Update coverage visualization
    img = coverage_to_image(covered)
    world_img.set_data(img)
    shared_img.set_data(img)
    
    # Update pheromone visualization
    pher_rgba = pheromone_to_rgba(pheromone)
    pher_img.set_data(pher_rgba)
    
    # Update robot positions
    positions = np.array([[r.x + 0.5, r.y + 0.5] for r in robots])
    robot_scatter.set_offsets(positions)
    
    # Color robots: black for active, red for failed
    colors = ['red' if r.failed else 'k' for r in robots]
    robot_scatter.set_facecolors(colors)
    robot_scatter.set_edgecolors(colors)
    
    # Update robot labels
    for i, r in enumerate(robots):
        robot_labels[i].set_position((r.x + 0.6, r.y + 0.6))
        robot_labels[i].set_text(f"R{r.id}" + (" (F)" if r.failed else ""))
        robot_labels[i].set_color('red' if r.failed else 'k')
    
    # Update discovered/undiscovered targets
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
    
    # Update title
    ax_world.set_title(
        "Stigmergy Search (Efficient) — Repulsive Pheromone\n"
        f"Covered: {covered.sum()} / {W*H} cells, Found: {len(found_targets)} / {len(targets)}"
    )
    
    # Save plot when all robots finish or targets found
    if not plot_saved and len(found_targets) >= len(targets):
        output_dir.mkdir(parents=True, exist_ok=True)
        save_targets_over_time_plot(output_dir / "targets_over_time.png", targets_found_over_time)
        
        if fail_robot_id is not None:
            np.save('output_metrics/stigmergy_search_efficient_with_failure.npy', np.array(targets_found_over_time))
        else:
            np.save('output_metrics/stigmergy_search_efficient_without_failure.npy', np.array(targets_found_over_time))
        plot_saved = True
        state_dict['plot_saved'] = plot_saved
    
    frame_writer.save(fig)
    return (world_img, pher_img, shared_img, robot_scatter, disc_plot, und_plot, *robot_labels)


if __name__ == "__main__":
    # Simulation Parameters
    GRID_SIZE = 100
    N_ROBOTS = 5
    N_TARGETS = 25
    STEPS_PER_FRAME = 5
    INTERVAL_MS = 50
    RANDOM_SEED = 42
    ROBOT_RADIUS = 5
    COLLISION_RADIUS = 1
    
    # Pheromone Parameters
    PHER_DEPOSIT = 1.0
    TAU_DECAY = 200.0
    PHER_MIN = 1e-6
    
    # Failure Parameters (set to None for no failure)
    FAIL_ROBOT_ID = None  # Set to robot ID (0 to N_ROBOTS-1) to trigger failure
    FAIL_AT_STEP = None   # Set to step number to trigger failure
    
    # Output
    OUTPUT_DIR = Path('output_frames/stigmergy_search_efficient')
    targets_found_over_time = []
    plot_saved = False
    global_step = 0
    failure_triggered = False
    
    # Initialize RNG and writer
    rng = np.random.default_rng(RANDOM_SEED)
    FPS = compute_fps(INTERVAL_MS)
    frame_writer = FrameWriter(str(OUTPUT_DIR))
    
    # Initialize grid
    W = H = GRID_SIZE
    
    # Generate random starting positions for robots
    start_positions = []
    for i in range(N_ROBOTS):
        x = rng.integers(0, W)
        y = rng.integers(0, H)
        start_positions.append((x, y))
    
    robots = [Robot(
        i,
        start_positions[i][0],
        start_positions[i][1],
        local_covered=np.zeros((H, W), dtype=bool)
    ) for i in range(N_ROBOTS)]
    
    # Initialize targets and coverage
    covered = np.zeros((H, W), dtype=bool)
    targets = generate_unique_targets(GRID_SIZE, N_TARGETS, rng)
    found_targets: Set[Tuple[int, int]] = set()
    pheromone = np.zeros((H, W), dtype=float)
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(16.0, 5.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.15)
    ax_world = fig.add_subplot(gs[0, 0])
    ax_pher = fig.add_subplot(gs[0, 1])
    ax_shared = fig.add_subplot(gs[0, 2])
    
    # --- World Map (Coverage) ---
    world_img = ax_world.imshow(coverage_to_image(covered), origin='lower', 
                                extent=[0, W, 0, H], vmin=0.0, vmax=1.0, cmap='gray')
    
    ax_world.set_title("World — Coverage Map")
    ax_world.set_xlim(0, W)
    ax_world.set_ylim(0, H)
    ax_world.set_aspect('equal', adjustable='box')
    ax_world.set_xticks(np.arange(0, W + 1, 20))
    ax_world.set_yticks(np.arange(0, H + 1, 20))
    ax_world.grid(which='major', color='k', alpha=0.15, linewidth=0.5)
    
    # Robot scatter
    robot_colors = ['k' for _ in robots]
    robot_scatter = ax_world.scatter([r.x + 0.5 for r in robots],
                                     [r.y + 0.5 for r in robots],
                                     s=40, marker='o', c=robot_colors, zorder=4)
    
    # Robot labels
    robot_labels = []
    for r in robots:
        txt = ax_world.text(r.x + 0.6, r.y + 0.6, f"R{r.id}", 
                           fontsize=7, color='k', zorder=6)
        robot_labels.append(txt)
    
    # Targets on world map
    if targets:
        tx_world, ty_world = zip(*targets)
    else:
        tx_world, ty_world = [], []
    ax_world.scatter([x + 0.5 for x in tx_world], [y + 0.5 for y in ty_world],
                    s=20, marker='x', c='r', alpha=0.9, zorder=4)
    
    # --- Pheromone Map ---
    pher_img = ax_pher.imshow(pheromone_to_rgba(pheromone), origin='lower',
                             extent=[0, W, 0, H])
    
    ax_pher.set_title("Pheromone Field (Repulsive)")
    ax_pher.set_xlim(0, W)
    ax_pher.set_ylim(0, H)
    ax_pher.set_aspect('equal', adjustable='box')
    ax_pher.set_xticks(np.arange(0, W + 1, 20))
    ax_pher.set_yticks(np.arange(0, H + 1, 20))
    ax_pher.grid(which='major', color='k', alpha=0.15, linewidth=0.5)
    
    # --- Shared Map (Targets) ---
    shared_img = ax_shared.imshow(coverage_to_image(covered), origin='lower',
                                  extent=[0, W, 0, H], vmin=0.0, vmax=1.0, cmap='gray')
    
    ax_shared.set_title("Target Discovery")
    ax_shared.set_xlim(0, W)
    ax_shared.set_ylim(0, H)
    ax_shared.set_aspect('equal', adjustable='box')
    ax_shared.set_xticks(np.arange(0, W + 1, 20))
    ax_shared.set_yticks(np.arange(0, H + 1, 20))
    ax_shared.grid(which='major', color='k', alpha=0.15, linewidth=0.5)
    
    # Target plots
    disc_plot = ax_shared.scatter([], [], s=25, marker='o',
                                 facecolors='none', edgecolors='g', 
                                 linewidths=1.5, label='Discovered')
    und_tx, und_ty = (list(zip(*targets)) if targets else ([], []))
    und_plot = ax_shared.scatter([x + 0.5 for x in und_tx], [y + 0.5 for y in und_ty],
                                s=18, marker='x', c='r', label='Undiscovered')
    ax_shared.legend(loc='upper right', fontsize=8, frameon=False)
    
    # State dictionary for mutable state
    state_dict = {
        'global_step': global_step,
        'failure_triggered': failure_triggered,
        'plot_saved': plot_saved
    }
    
    def update_wrapper(frame):
        return update(frame, robots, pheromone, covered, targets, found_targets, W, H,
                     state_dict['global_step'], FAIL_AT_STEP, FAIL_ROBOT_ID,
                     state_dict['failure_triggered'], targets_found_over_time,
                     world_img, pher_img, shared_img, robot_scatter, robot_labels,
                     disc_plot, und_plot, ax_world, fig, frame_writer, STEPS_PER_FRAME,
                     OUTPUT_DIR, state_dict['plot_saved'], ROBOT_RADIUS, COLLISION_RADIUS,
                     PHER_DEPOSIT, TAU_DECAY, PHER_MIN, state_dict)
    
    anim = run_animation(fig, update_wrapper, frames=2000000, interval_ms=INTERVAL_MS, blit=False)
    plt.show()
