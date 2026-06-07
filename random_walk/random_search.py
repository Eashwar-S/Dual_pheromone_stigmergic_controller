import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Set, Tuple
import sys
CURRENT_DIR = Path(__file__).resolve().parent          
PROJECT_ROOT = CURRENT_DIR.parent                     
sys.path.insert(0, str(PROJECT_ROOT))


from common.simulation import FrameWriter, compute_fps, make_writer, run_animation
from common.utilities import generate_unique_targets, mark_visible, discover_targets_in_vnhood
from common.visualization import coverage_to_image, save_targets_over_time_plot, plot_visit_counts
from robot_random import Robot


def maybe_trigger_failure(global_step, fail_at_step, fail_robot_id, robots, failure_triggered):
    """Trigger failure at specified step."""
    if failure_triggered or fail_robot_id is None or fail_at_step is None:
        return failure_triggered
    if global_step == fail_at_step:
        robots[fail_robot_id].failed = True
        return True
    return failure_triggered


def sim_step(robots, covered_global, targets, found_targets, W, H, rng, global_step, fail_at_step,
             fail_robot_id, failure_triggered, targets_found_over_time, collision_radius):
    """Execute one simulation step."""
    failure_triggered = maybe_trigger_failure(global_step, fail_at_step, fail_robot_id, robots, failure_triggered)

    for r in robots:
        if r.failed:
            continue
        mark_visible(r.local_covered, r.x, r.y, ROBOT_RADIUS)
        mark_visible(covered_global, r.x, r.y, ROBOT_RADIUS)
        discover_targets_in_vnhood(r.x, r.y, targets, found_targets, W, H, ROBOT_RADIUS)

    for r in robots:
        if r.failed:
            continue

        old_x, old_y = r.x, r.y
        r.step(rng, robots)

        # Count visit only once after movement
        if (r.x, r.y) != (old_x, old_y):
            mark_visited_counts(visit_counts, r.x, r.y, ROBOT_RADIUS)

        mark_visible(r.local_covered, r.x, r.y, ROBOT_RADIUS)
        mark_visible(covered_global, r.x, r.y, ROBOT_RADIUS)
        discover_targets_in_vnhood(r.x, r.y, targets, found_targets, W, H, ROBOT_RADIUS)

    
    global_step += 1
    targets_found_over_time.append(len(found_targets))
    
    if len(found_targets) >= len(targets):
        print(f"\nAll targets discovered at step {global_step}")
        sys.exit(0)
    
    return global_step, failure_triggered


def update(frame, robots, covered_global, targets, found_targets, W, H, rng, global_step, fail_at_step,
           fail_robot_id, failure_triggered, targets_found_over_time, world_cov_img,
           robot_scat, robot_labels, obs_cov_img, disc_plot, und_plot, ax_world, fig,
           frame_writer, steps_per_frame, output_dir, plot_saved, collision_radius, state_dict):
    """Animation update function."""
    for _ in range(steps_per_frame):
        global_step, failure_triggered = sim_step(
            robots, covered_global, targets, found_targets, W, H, rng, global_step, fail_at_step,
            fail_robot_id, failure_triggered, targets_found_over_time, collision_radius
        )
    
    state_dict['global_step'] = global_step
    state_dict['failure_triggered'] = failure_triggered
    
    coverage_img = coverage_to_image(covered_global)
    world_cov_img.set_data(coverage_img)
    robot_scat.set_offsets(np.array([[r.x + 0.5, r.y + 0.5] for r in robots]))
    obs_cov_img.set_data(coverage_img)
    
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
        "World - Memoryless Random Walk\n"
        f"Covered (union): {covered_global.sum()} / {W*H}, Found targets: {len(found_targets)} / {len(targets)}"
    )
    
    if (not plot_saved) and len(found_targets) >= len(targets):
        output_dir.mkdir(parents=True, exist_ok=True)
        save_targets_over_time_plot(output_dir / "targets_over_time_random_walk.png", targets_found_over_time)
        
        if fail_robot_id is not None:
            np.save('output_metrics/random_walk_with_failure.npy', np.array(targets_found_over_time))
        else:
            np.save('output_metrics/random_walk_without_failure.npy', np.array(targets_found_over_time))
        plot_saved = True
        state_dict['plot_saved'] = plot_saved
    
    frame_writer.save(fig)
    return (world_cov_img, robot_scat, obs_cov_img, und_plot, disc_plot, *robot_labels)



def mark_visited_counts(visit_counts: np.ndarray, x: int, y: int, radius: int):
    """Increment visit count for the robot cell and surrounding cells within radius."""
    H, W = visit_counts.shape

    x_min = max(0, x - radius)
    x_max = min(W - 1, x + radius)
    y_min = max(0, y - radius)
    y_max = min(H - 1, y + radius)

    for yy in range(y_min, y_max + 1):
        for xx in range(x_min, x_max + 1):
            visit_counts[yy, xx] += 1



def run_headless_random_walk(
    robots,
    covered_global,
    visit_counts,
    targets,
    found_targets,
    W,
    H,
    rng,
    max_steps,
    robot_radius,
    fail_robot_id=None,
    fail_at_step=None,
):
    """
    Run memoryless random walk without live animation.
    """
    targets_found_over_time = []
    failure_triggered = False

    for step in range(max_steps):


        if (
            not failure_triggered
            and fail_robot_id is not None
            and fail_at_step is not None
            and step == fail_at_step
        ):
            robots[fail_robot_id].failed = True
            failure_triggered = True

        for r in robots:
            if r.failed:
                continue

            mark_visible(covered_global, r.x, r.y, robot_radius)
            mark_visible(r.local_covered, r.x, r.y, robot_radius)
            discover_targets_in_vnhood(r.x, r.y, targets, found_targets, W, H, robot_radius)


        for r in robots:
            if r.failed:
                continue

            old_x, old_y = r.x, r.y
            r.step(rng, robots)

            if (r.x, r.y) != (old_x, old_y):
                mark_visited_counts(visit_counts, r.x, r.y, ROBOT_RADIUS)

            mark_visible(covered_global, r.x, r.y, robot_radius)
            mark_visible(r.local_covered, r.x, r.y, robot_radius)
            discover_targets_in_vnhood(r.x, r.y, targets, found_targets, W, H, robot_radius)

        targets_found_over_time.append(len(found_targets))

        if len(found_targets) >= len(targets):
            print(f"\nAll targets discovered at step {step + 1}")
            break

    return {
        "steps": step + 1,
        "n_targets_found": len(found_targets),
        "targets_found_over_time": targets_found_over_time,
        "failure_triggered": failure_triggered,
    }


if __name__ == "__main__":
    GRID_SIZE = 100
    N_ROBOTS = 5
    N_TARGETS = 40
    RANDOM_SEED = 7
    ROBOT_RADIUS = 1

    STEPS_PER_FRAME = 10
    INTERVAL_MS = 50
    MAX_STEPS = 20000

    VISUALIZATION_MODE = "final"
    # Options:
    # "none"  -> fastest, no plotting
    # "final" -> fast simulation + one final visit-count plot
    # "live"  -> live animation

    SAVE_FINAL_PLOT = True

    targets_found_over_time = []
    plot_saved = False
    OUTPUT_DIR = Path("output_frames/random_walk/")

    FAIL_ROBOT_ID = None
    FAIL_AT_STEP = None

    global_step = 0
    failure_triggered = False

    COLLISION_RADIUS = 1

    rng = np.random.default_rng(RANDOM_SEED)

    W = H = GRID_SIZE

    start_positions = []
    for i in range(N_ROBOTS):
        x = rng.integers(0, W)
        y = rng.integers(0, H)
        start_positions.append((x, y))

    robots = [
        Robot(
            id=i,
            x=start_positions[i][0],
            y=start_positions[i][1],
            robot_radius=ROBOT_RADIUS,
            collision_radius=COLLISION_RADIUS,
            local_covered=np.zeros((H, W), dtype=bool),
        )
        for i in range(N_ROBOTS)
    ]

    targets = generate_unique_targets(GRID_SIZE, N_TARGETS, rng)
    found_targets: Set[Tuple[int, int]] = set()

    covered_global = np.zeros((H, W), dtype=bool)
    visit_counts = np.zeros((H, W), dtype=int)

    if VISUALIZATION_MODE in ["none", "final"]:
        result = run_headless_random_walk(
            robots=robots,
            covered_global=covered_global,
            visit_counts=visit_counts,
            targets=targets,
            found_targets=found_targets,
            W=W,
            H=H,
            rng=rng,
            max_steps=MAX_STEPS,
            robot_radius=ROBOT_RADIUS,
            fail_robot_id=FAIL_ROBOT_ID,
            fail_at_step=FAIL_AT_STEP,
        )

        print(
            f"\nFinished in {result['steps']} steps. "
            f"Found targets: {result['n_targets_found']} / {len(targets)}"
        )

        if VISUALIZATION_MODE == "final":
            final_plot_path = None

            if SAVE_FINAL_PLOT:
                final_plot_path = OUTPUT_DIR / "final_visit_counts.png"

            plot_visit_counts(
                visit_counts=visit_counts,
                robots=robots,
                targets=targets,
                found_targets=found_targets,
                output_path=final_plot_path,
            )

    elif VISUALIZATION_MODE == "live":
        FPS = compute_fps(INTERVAL_MS)
        writer = make_writer(INTERVAL_MS, title="Random walk", artist="you")
        frame_writer = FrameWriter("output_frames/random_walk/")

        fig = plt.figure(figsize=(12.5, 6.2))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.12)

        ax_world = fig.add_subplot(gs[0, 0])
        ax_obs = fig.add_subplot(gs[0, 1])

        world_cov_img = ax_world.imshow(
            coverage_to_image(covered_global),
            origin="lower",
            extent=[0, W, 0, H],
            vmin=0.0,
            vmax=1.0,
            zorder=0,
        )

        ax_world.set_xticks(np.arange(0, W + 1, 10))
        ax_world.set_yticks(np.arange(0, H + 1, 10))
        ax_world.set_xticks(np.arange(0, W + 1, 1), minor=True)
        ax_world.set_yticks(np.arange(0, H + 1, 1), minor=True)
        ax_world.grid(which="major", color="k", alpha=0.15, linewidth=0.5)
        ax_world.grid(which="minor", color="k", alpha=0.05, linewidth=0.2)

        robot_colors = ["k" for _ in robots]
        robot_scat = ax_world.scatter(
            [r.x + 0.5 for r in robots],
            [r.y + 0.5 for r in robots],
            s=40,
            marker="o",
            c=robot_colors,
            zorder=3,
        )

        robot_labels = []
        for r in robots:
            t = ax_world.text(
                r.x + 0.6,
                r.y + 0.6,
                f"R{r.id}",
                fontsize=7,
                color="k",
                zorder=5,
            )
            robot_labels.append(t)

        if targets:
            tx, ty = zip(*targets)
        else:
            tx, ty = [], []

        ax_world.scatter(
            [x + 0.5 for x in tx],
            [y + 0.5 for y in ty],
            s=20,
            marker="x",
            c="r",
            alpha=0.9,
            zorder=4,
        )

        ax_world.set_title("World - Memoryless Random Walk")
        ax_world.set_xlim(0, W)
        ax_world.set_ylim(0, H)
        ax_world.set_aspect("equal", adjustable="box")

        obs_cov_img = ax_obs.imshow(
            coverage_to_image(covered_global),
            origin="lower",
            extent=[0, W, 0, H],
            zorder=0,
        )

        und_plot = ax_obs.scatter(
            [x + 0.5 for x in tx],
            [y + 0.5 for y in ty],
            s=18,
            marker="x",
            c="r",
            label="Undiscovered",
            zorder=2,
        )

        disc_plot = ax_obs.scatter(
            [],
            [],
            s=25,
            marker="o",
            facecolors="none",
            edgecolors="g",
            linewidths=1.5,
            label="Discovered",
            zorder=2,
        )

        ax_obs.set_xticks(np.arange(0, W + 1, 10))
        ax_obs.set_yticks(np.arange(0, H + 1, 10))
        ax_obs.set_xticks(np.arange(0, W + 1, 1), minor=True)
        ax_obs.set_yticks(np.arange(0, H + 1, 1), minor=True)
        ax_obs.grid(which="major", color="k", alpha=0.15, linewidth=0.5)
        ax_obs.grid(which="minor", color="k", alpha=0.05, linewidth=0.2)
        ax_obs.set_title("Observer - Target Discovery")
        ax_obs.set_xlim(0, W)
        ax_obs.set_ylim(0, H)
        ax_obs.set_aspect("equal", adjustable="box")
        ax_obs.legend(loc="upper right", fontsize=8, frameon=False)

        state_dict = {
            "global_step": global_step,
            "failure_triggered": failure_triggered,
            "plot_saved": plot_saved,
        }

        def update_wrapper(frame):
            return update(
                frame,
                robots,
                covered_global,
                targets,
                found_targets,
                W,
                H,
                rng,
                state_dict["global_step"],
                FAIL_AT_STEP,
                FAIL_ROBOT_ID,
                state_dict["failure_triggered"],
                targets_found_over_time,
                world_cov_img,
                robot_scat,
                robot_labels,
                obs_cov_img,
                disc_plot,
                und_plot,
                ax_world,
                fig,
                frame_writer,
                STEPS_PER_FRAME,
                OUTPUT_DIR,
                state_dict["plot_saved"],
                COLLISION_RADIUS,
                state_dict,
            )

        anim = run_animation(
            fig,
            update_wrapper,
            frames=MAX_STEPS,
            interval_ms=INTERVAL_MS,
            blit=False,
        )

        plt.show()

    else:
        raise ValueError(
            "Invalid VISUALIZATION_MODE. Use 'none', 'final', or 'live'."
        )
    
    
    
    
    
    
    
