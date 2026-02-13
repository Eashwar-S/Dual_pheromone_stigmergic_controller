import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Set, Tuple
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common.simulation import FrameWriter, compute_fps, make_writer, run_animation
from src.common.utilities import generate_unique_targets, mark_visible, discover_targets_in_vnhood
from src.common.visualization import coverage_to_image, pheromone_to_rgba, save_targets_over_time_plot
from src.stigmergy.pheromone import apply_decay
from src.stigmergy.robot_random import Robot


def maybe_trigger_failure(global_step, fail_at_step, fail_robot_id, robots, failure_triggered):
    """Trigger failure at specified step."""
    if failure_triggered or fail_robot_id is None or fail_at_step is None:
        return failure_triggered
    if global_step == fail_at_step:
        robots[fail_robot_id].failed = True
        return True
    return failure_triggered


def sim_step(robots, pher, covered_global, targets, found_targets, W, H, tau_decay, pher_min,
             pher_deposit, bias_alpha, uncovered_bonus, rng, global_step, fail_at_step,
             fail_robot_id, failure_triggered, targets_found_over_time, collision_radius):
    """Execute one simulation step."""
    failure_triggered = maybe_trigger_failure(global_step, fail_at_step, fail_robot_id, robots, failure_triggered)
    
    apply_decay(pher, tau_decay, pher_min)
    
    for r in robots:
        mark_visible(r.local_covered, r.x, r.y)
        mark_visible(covered_global, r.x, r.y)
        discover_targets_in_vnhood(r.x, r.y, targets, found_targets, W, H)
        r.deposit_pheromone(pher, pher_deposit, r=5)
    
    for r in robots:
        r.step(pher, bias_alpha, uncovered_bonus, rng, robots, collision_radius)
    
    for r in robots:
        mark_visible(r.local_covered, r.x, r.y)
        mark_visible(covered_global, r.x, r.y)
        discover_targets_in_vnhood(r.x, r.y, targets, found_targets, W, H)
        r.deposit_pheromone(pher, 0.3 * pher_deposit, r=5)
    
    global_step += 1
    targets_found_over_time.append(len(found_targets))
    
    if len(found_targets) >= len(targets):
        print(f"\nAll targets discovered at step {global_step}")
        sys.exit(0)
    
    return global_step, failure_triggered


def update(frame, robots, pher, covered_global, targets, found_targets, W, H, tau_decay, pher_min,
           pher_deposit, bias_alpha, uncovered_bonus, rng, global_step, fail_at_step,
           fail_robot_id, failure_triggered, targets_found_over_time, world_cov_img, world_pher_img,
           robot_scat, robot_labels, obs_pher_img, disc_plot, und_plot, ax_world, fig,
           frame_writer, steps_per_frame, output_dir, plot_saved, collision_radius, state_dict):
    """Animation update function."""
    for _ in range(steps_per_frame):
        global_step, failure_triggered = sim_step(
            robots, pher, covered_global, targets, found_targets, W, H, tau_decay, pher_min,
            pher_deposit, bias_alpha, uncovered_bonus, rng, global_step, fail_at_step,
            fail_robot_id, failure_triggered, targets_found_over_time, collision_radius
        )
    
    state_dict['global_step'] = global_step
    state_dict['failure_triggered'] = failure_triggered
    
    world_cov_img.set_data(coverage_to_image(covered_global))
    world_pher_img.set_data(pheromone_to_rgba(pher))
    robot_scat.set_offsets(np.array([[r.x + 0.5, r.y + 0.5] for r in robots]))
    obs_pher_img.set_data(pheromone_to_rgba(pher))
    
    colors = ['red' if r.failed else 'k' for r in robots]
    robot_scat.set_facecolors(colors)
    robot_scat.set_edgecolors(colors)
    for i, r in enumerate(robots):
        robot_labels[i].set_position((r.x + 0.6, r.y + 0.6))
        robot_labels[i].set_text(f"R{r.id}" + (" (failed)" if r.failed else ""))
        robot_labels[i].set_color('red' if r.failed else 'k')
    
    discovered = list(found_targets)
    undiscovered = list(targets - found_targets)
    if discovered:
        dx, dy = zip(*discovered)
        disc_plot.set_offsets(np.c_[[x + 0.5 for x in dx], [y + 0.5 for y in dy]])
    else:
        disc_plot.set_offsets(np.empty((0, 2)))
    if undiscovered:
        ux, uy = zip(*undiscovered)
        und_plot.set_offsets(np.c_[[x + 0.5 for x in ux], [y + 0.5 for y in uy]])
    else:
        und_plot.set_offsets(np.empty((0, 2)))
    
    ax_world.set_title(
        "World — Stigmergy (Local Maps + Random Walk + Pheromone)\n"
        f"Covered (union): {covered_global.sum()} / {W*H}, Found targets: {len(found_targets)} / {len(targets)}"
    )
    
    if (not plot_saved) and len(found_targets) >= len(targets):
        output_dir.mkdir(parents=True, exist_ok=True)
        save_targets_over_time_plot(output_dir / "targets_over_time_stigmergy.png", targets_found_over_time)
        
        if fail_robot_id is not None:
            np.save('output_metrics/stigmergy_with_failure.npy', np.array(targets_found_over_time))
        else:
            np.save('output_metrics/stigmergy_without_failure.npy', np.array(targets_found_over_time))
        plot_saved = True
        state_dict['plot_saved'] = plot_saved
    
    frame_writer.save(fig)
    return (world_cov_img, world_pher_img, robot_scat, obs_pher_img, und_plot, disc_plot, *robot_labels)


if __name__ == "__main__":
    GRID_SIZE = 100
    N_ROBOTS = 10
    N_TARGETS = 5
    RANDOM_SEED = 7
    STEPS_PER_FRAME = 10
    INTERVAL_MS = 50
    
    targets_found_over_time = []
    plot_saved = False
    OUTPUT_DIR = Path("output_frames/stigmergy_random_walk/")
    
    FAIL_ROBOT_ID = None
    FAIL_AT_STEP = None
    global_step = 0
    failure_triggered = False
    
    PHER_DEPOSIT = 1.0
    TAU_DECAY = 600.0
    PHER_MIN = 1e-6
    BIAS_ALPHA = 250
    UNCOVERED_BONUS = 10.0
    COLLISION_RADIUS = 1  # 1 = 3x3 block safe zone
    rng = np.random.default_rng(RANDOM_SEED)
    
    FPS = compute_fps(INTERVAL_MS)
    writer = make_writer(INTERVAL_MS, title="Stigmergy - random walk", artist="you")
    frame_writer = FrameWriter("output_frames/stigmergy_random_walk/")
    
    W = H = GRID_SIZE
    pts = rng.random((N_ROBOTS, 2)) * np.array([W, H])
    robots = [Robot(i, int(pts[i, 0]), int(pts[i, 1]), local_covered=np.zeros((H, W), dtype=bool)) 
              for i in range(N_ROBOTS)]
    
    targets = generate_unique_targets(GRID_SIZE, N_TARGETS, rng)
    found_targets: Set[Tuple[int, int]] = set()
    
    covered_global = np.zeros((H, W), dtype=bool)
    pher = np.zeros((H, W), dtype=float)
    
    fig = plt.figure(figsize=(12.5, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.12)
    ax_world = fig.add_subplot(gs[0, 0])
    ax_obs = fig.add_subplot(gs[0, 1])
    
    world_cov_img = ax_world.imshow(coverage_to_image(covered_global), origin='lower',
                                    extent=[0, W, 0, H], vmin=0.0, vmax=1.0, zorder=0)
    world_pher_img = ax_world.imshow(pheromone_to_rgba(pher), origin='lower',
                                     extent=[0, W, 0, H], zorder=1)
    
    ax_world.set_xticks(np.arange(0, W + 1, 10))
    ax_world.set_yticks(np.arange(0, H + 1, 10))
    ax_world.set_xticks(np.arange(0, W + 1, 1), minor=True)
    ax_world.set_yticks(np.arange(0, H + 1, 1), minor=True)
    ax_world.grid(which='major', color='k', alpha=0.15, linewidth=0.5)
    ax_world.grid(which='minor', color='k', alpha=0.05, linewidth=0.2)
    
    robot_colors = ['k' for _ in robots]
    robot_scat = ax_world.scatter([r.x + 0.5 for r in robots],
                                  [r.y + 0.5 for r in robots],
                                  s=40, marker='o', c=robot_colors, zorder=3)
    
    robot_labels = []
    for r in robots:
        t = ax_world.text(r.x + 0.6, r.y + 0.6, f"R{r.id}",
                         fontsize=7, color='k', zorder=5)
        robot_labels.append(t)
    
    if targets:
        tx, ty = zip(*targets)
    else:
        tx, ty = [], []
    ax_world.scatter([x + 0.5 for x in tx], [y + 0.5 for y in ty],
                    s=20, marker='x', c='r', alpha=0.9, zorder=4)
    
    ax_world.set_title("World — Stigmergy (Local Maps + Random Walk + Pheromone)")
    ax_world.set_xlim(0, W)
    ax_world.set_ylim(0, H)
    ax_world.set_aspect('equal', adjustable='box')
    
    obs_pher_img = ax_obs.imshow(pheromone_to_rgba(pher), origin='lower',
                                extent=[0, W, 0, H], zorder=0)
    und_plot = ax_obs.scatter([x + 0.5 for x in tx], [y + 0.5 for y in ty],
                             s=18, marker='x', c='r', label='Undiscovered', zorder=2)
    disc_plot = ax_obs.scatter([], [], s=25, marker='o', facecolors='none',
                              edgecolors='g', linewidths=1.5, label='Discovered', zorder=2)
    
    ax_obs.set_xticks(np.arange(0, W + 1, 10))
    ax_obs.set_yticks(np.arange(0, H + 1, 10))
    ax_obs.set_xticks(np.arange(0, W + 1, 1), minor=True)
    ax_obs.set_yticks(np.arange(0, H + 1, 1), minor=True)
    ax_obs.grid(which='major', color='k', alpha=0.15, linewidth=0.5)
    ax_obs.grid(which='minor', color='k', alpha=0.05, linewidth=0.2)
    ax_obs.set_title("Observer — Pheromone Field (Robots don't share maps)")
    ax_obs.set_xlim(0, W)
    ax_obs.set_ylim(0, H)
    ax_obs.set_aspect('equal', adjustable='box')
    ax_obs.legend(loc='upper right', fontsize=8, frameon=False)
    
    state_dict = {
        'global_step': global_step,
        'failure_triggered': failure_triggered,
        'plot_saved': plot_saved
    }
    
    def update_wrapper(frame):
        return update(frame, robots, pher, covered_global, targets, found_targets, W, H, TAU_DECAY, PHER_MIN,
                     PHER_DEPOSIT, BIAS_ALPHA, UNCOVERED_BONUS, rng, state_dict['global_step'], FAIL_AT_STEP,
                     FAIL_ROBOT_ID, state_dict['failure_triggered'], targets_found_over_time, world_cov_img, world_pher_img,
                     robot_scat, robot_labels, obs_pher_img, disc_plot, und_plot, ax_world, fig,
                     frame_writer, STEPS_PER_FRAME, OUTPUT_DIR, state_dict['plot_saved'], COLLISION_RADIUS, state_dict)
    
    anim = run_animation(fig, update_wrapper, frames=2000, interval_ms=INTERVAL_MS, blit=False)
    plt.show()
