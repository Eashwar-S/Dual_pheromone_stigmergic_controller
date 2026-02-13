import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Set, Tuple, Dict
from collections import deque
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common.simulation import FrameWriter, compute_fps, run_animation
from src.common.utilities import generate_unique_targets, mark_visible, discover_targets_in_vnhood
from src.common.visualization import coverage_to_image, combined_pheromone_to_rgba
from src.stigmergy.pheromone import apply_decay
from src.stigmergy.robot_spiral import Robot


def sim_step(robots, pher_repulse, pher_attract, covered_global, targets, found_targets,
             W, H, tau_decay, pher_min, global_target_visits, collision_radius, pher_deposit, 
             metrics_log, R0_data, R1_data):
    """Execute one simulation step with utility term tracking."""
    apply_decay(pher_repulse, tau_decay, pher_min)
    
    pher_repulse[pher_repulse < 1e-5] = 0.0
    pher_attract[pher_attract < 1e-5] = 0.0
    
    for r in robots:
        mark_visible(r.local_covered, r.x, r.y)
        mark_visible(covered_global, r.x, r.y)
        discover_targets_in_vnhood(r.x, r.y, targets, found_targets, W, H)
    
    for r in robots:
        r.step(pher_repulse, pher_attract, targets, robots, global_target_visits, collision_radius, pher_deposit)
    
    for r_idx, r in enumerate(robots):
        if r_idx == 0:
            R0_data['t1'].append(r.current_terms[0])
            R0_data['t2'].append(r.current_terms[1])
            R0_data['t3'].append(r.current_terms[2])
        elif r_idx == 1:
            R1_data['t1'].append(r.current_terms[0])
            R1_data['t2'].append(r.current_terms[1])
            R1_data['t3'].append(r.current_terms[2])
    
    all_targets_satisfied = False
    if len(targets) > 0:
        met_targets = 0
        for t in targets:
            visitors = global_target_visits.get(t, set())
            if len(visitors) > 1:
                met_targets += 1
        if met_targets == len(targets):
            all_targets_satisfied = True
    
    targets_satisfied_count = sum(1 for t in targets if len(global_target_visits.get(t, set())) > 1)
    metrics_log.append((len(found_targets), targets_satisfied_count))
    
    if all_targets_satisfied:
        print(f"\nSUCCESS: All targets visited by >1 robot")
        sys.exit(0)


def update(frame, robots, pher_repulse, pher_attract, covered_global, targets, found_targets,
           W, H, tau_decay, pher_min, global_target_visits, collision_radius, pher_deposit,
           metrics_log, world_cov_img, world_pher_img, robot_scat, robot_labels, obs_pher_img,
           disc_plot, ax_world, fig, frame_writer, steps_per_frame, global_step, state_dict,
           R0_data, R1_data):#, lines0, lines1, ax_r0, ax_r1):
    """Animation update function with utility term plotting."""
    for _ in range(steps_per_frame):
        sim_step(robots, pher_repulse, pher_attract, covered_global, targets, found_targets,
                W, H, tau_decay, pher_min, global_target_visits, collision_radius, pher_deposit, 
                metrics_log, R0_data, R1_data)
        global_step += 1
    
    state_dict['global_step'] = global_step
    
    world_cov_img.set_data(coverage_to_image(covered_global))
    combined_rgba = combined_pheromone_to_rgba(pher_repulse, pher_attract)
    world_pher_img.set_data(combined_rgba)
    obs_pher_img.set_data(combined_rgba)
    
    offsets = np.array([[r.x + 0.5, r.y + 0.5] for r in robots])
    robot_scat.set_offsets(offsets)
    
    colors = []
    for r in robots:
        if r.mode == "SEARCH":
            colors.append('k')
        elif r.mode == "ADVERTISE":
            colors.append('blue')
        elif r.mode == "FOLLOW":
            colors.append('purple')
    
    robot_scat.set_facecolors(colors)
    robot_scat.set_edgecolors(colors)
    
    for i, r in enumerate(robots):
        robot_labels[i].set_position((r.x + 0.6, r.y + 0.6))
        robot_labels[i].set_text(f"R{r.id}")
    
    if targets:
        tx, ty = zip(*targets)
        t_colors = []
        for t in targets:
            visitors = global_target_visits.get(t, set())
            if len(visitors) > 1:
                t_colors.append('blue')
            elif len(visitors) == 1:
                t_colors.append('green')
            else:
                t_colors.append('red')
        
        ax_world.collections[1].remove()
        ax_world.scatter([x + 0.5 for x in tx], [y + 0.5 for y in ty],
                        s=40, marker='x', c=t_colors, linewidths=2, zorder=4)
    
    discovered = list(found_targets)
    if discovered:
        dx, dy = zip(*discovered)
        disc_plot.set_offsets(np.c_[[x + 0.5 for x in dx], [y + 0.5 for y in dy]])
    
    title = f"Step {global_step}: Targets Satisfied (>1 visitor): {metrics_log[-1][1]}/{len(targets)}"
    ax_world.set_title(title)
    
    # lines0[0].set_data(range(len(R0_data['t1'])), R0_data['t1'])
    # lines0[1].set_data(range(len(R0_data['t2'])), R0_data['t2'])
    # lines0[2].set_data(range(len(R0_data['t3'])), R0_data['t3'])
    # ax_r0.relim()
    # ax_r0.autoscale_view()
    
    # lines1[0].set_data(range(len(R1_data['t1'])), R1_data['t1'])
    # lines1[1].set_data(range(len(R1_data['t2'])), R1_data['t2'])
    # lines1[2].set_data(range(len(R1_data['t3'])), R1_data['t3'])
    # ax_r1.relim()
    # ax_r1.autoscale_view()
    
    frame_writer.save(fig)
    return (world_cov_img, world_pher_img, robot_scat)#, *lines0, *lines1)


if __name__ == "__main__":
    GRID_SIZE = 100
    N_TARGETS = 1
    N_ROBOTS = 2
    
    RANDOM_SEED = 42
    STEPS_PER_FRAME = 15
    INTERVAL_MS = 50
    COLLISION_RADIUS = 1
    
    OUTPUT_DIR = Path("output_frames/stigmergy_spiral_plotting/")
    metrics_log = []
    
    PLOT_LEN = 200
    R0_data = {'t1': deque(maxlen=PLOT_LEN), 't2': deque(maxlen=PLOT_LEN), 't3': deque(maxlen=PLOT_LEN)}
    R1_data = {'t1': deque(maxlen=PLOT_LEN), 't2': deque(maxlen=PLOT_LEN), 't3': deque(maxlen=PLOT_LEN)}
    
    PHER_DEPOSIT = 1.0
    TAU_DECAY = 200.0
    PHER_MIN = 1e-6
    rng = np.random.default_rng(RANDOM_SEED)
    
    FPS = compute_fps(INTERVAL_MS)
    frame_writer = FrameWriter(str(OUTPUT_DIR))
    
    W = H = GRID_SIZE
    global_step = 0
    
    pts = np.array([[10, 10], [90, 90]])
    
    robots = [Robot(
        i,
        int(pts[i, 0]),
        int(pts[i, 1]),
        local_covered=np.zeros((H, W), dtype=bool),
        start_x=int(pts[i, 0]),
        start_y=int(pts[i, 1])
    ) for i in range(N_ROBOTS)]
    
    targets = {(50, 50)}
    found_targets = set()
    global_target_visits: Dict[Tuple[int, int], Set[int]] = {}
    
    covered_global = np.zeros((H, W), dtype=bool)
    pher_repulse = np.zeros((H, W), dtype=float)
    pher_attract = np.zeros((H, W), dtype=float)
    
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(1, 2)
    
    ax_world = fig.add_subplot(gs[0, 0])
    ax_obs = fig.add_subplot(gs[0, 1])
    # ax_r0 = fig.add_subplot(gs[1, 0])
    # ax_r1 = fig.add_subplot(gs[1, 1])
    
    world_cov_img = ax_world.imshow(coverage_to_image(covered_global), origin='lower',
                                    extent=[0, W, 0, H], vmin=0, vmax=1, zorder=0)
    world_pher_img = ax_world.imshow(combined_pheromone_to_rgba(pher_repulse, pher_attract),
                                     origin='lower', extent=[0, W, 0, H], zorder=1)
    
    robot_scat = ax_world.scatter([], [], s=40, zorder=5)
    tx, ty = zip(*targets)
    ax_world.scatter([x + 0.5 for x in tx], [y + 0.5 for y in ty], s=40, marker='x', c='r', zorder=4)
    
    robot_labels = [ax_world.text(0, 0, "", fontsize=7, color='white', fontweight='bold', zorder=6) 
                   for _ in robots]
    
    ax_world.set_xlim(0, W)
    ax_world.set_ylim(0, H)
    ax_world.set_title("World Map (Black=Trail, Blue=Target)")
    
    obs_pher_img = ax_obs.imshow(combined_pheromone_to_rgba(pher_repulse, pher_attract),
                                origin='lower', extent=[0, W, 0, H])
    disc_plot = ax_obs.scatter([], [], s=50, marker='o', edgecolors='g', facecolors='none')
    ax_obs.set_title("Pheromone Field (Detail)")
    
    # lines0 = []
    # lines0.append(ax_r0.plot([], [], label='T1 (Glb Expl)', color='red')[0])
    # lines0.append(ax_r0.plot([], [], label='T2 (Loc Expl)', color='green')[0])
    # lines0.append(ax_r0.plot([], [], label='T3 (Pher Rep)', color='blue')[0])
    # ax_r0.set_title("Robot 0 Utility Terms")
    # ax_r0.legend(loc='upper left', fontsize='small')
    # ax_r0.grid(True)
    
    # lines1 = []
    # lines1.append(ax_r1.plot([], [], label='T1 (Glb Expl)', color='red')[0])
    # lines1.append(ax_r1.plot([], [], label='T2 (Loc Expl)', color='green')[0])
    # lines1.append(ax_r1.plot([], [], label='T3 (Pher Rep)', color='blue')[0])
    # ax_r1.set_title("Robot 1 Utility Terms")
    # ax_r1.legend(loc='upper left', fontsize='small')
    # ax_r1.grid(True)
    
    plt.tight_layout()
    
    state_dict = {'global_step': global_step}
    
    def update_wrapper(frame):
        return update(frame, robots, pher_repulse, pher_attract, covered_global, targets, found_targets,
                     W, H, TAU_DECAY, PHER_MIN, global_target_visits, COLLISION_RADIUS, PHER_DEPOSIT,
                     metrics_log, world_cov_img, world_pher_img, robot_scat, robot_labels, obs_pher_img,
                     disc_plot, ax_world, fig, frame_writer, STEPS_PER_FRAME, state_dict['global_step'], 
                     state_dict, R0_data, R1_data)#, lines0, lines1, ax_r0, ax_r1)
    
    anim = run_animation(fig, update_wrapper, frames=1500, interval_ms=INTERVAL_MS, blit=False)
    plt.show()
