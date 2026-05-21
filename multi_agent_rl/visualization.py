import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
import sys
from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent       
PROJECT_ROOT = CURRENT_DIR.parent                
sys.path.insert(0, str(PROJECT_ROOT))
PROJECT_ROOT = CURRENT_DIR.parent              

sys.path.insert(0, str(CURRENT_DIR))
from common.visualization import coverage_to_image, plot_visit_counts

# Import your environment and network
from environment import MultiAgentGridEnv
from model import DQNPolicy

def coverage_to_image(cov):
    """Convert boolean coverage array to RGBA image (matches centralized_search)."""
    img = np.ones((cov.shape[0], cov.shape[1], 4))
    img[cov] = [0.85, 0.85, 0.85, 1.0] # Light gray for covered
    img[~cov] = [1.0, 1.0, 1.0, 1.0]   # White for uncovered
    return img

def mark_visible(covered, x, y, r=1):
    """Mark cells within radius r as covered."""
    H, W = covered.shape
    x_min, x_max = max(0, x - r), min(W - 1, x + r)
    y_min, y_max = max(0, y - r), min(H - 1, y + r)
    covered[x_min:x_max+1, y_min:y_max+1] = True # Note: env uses (x,y) indexing

def visualize_trained_policy(checkpoint_path="checkpoints/best_policy.pth", grid_size=20, num_agents=5, num_targets=0, num_obstacles=0):
    # Set up hardware and load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQNPolicy(input_channels=4, fov_size=3, action_space=4).to(device)
    
    try:
        policy_net.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        policy_net.eval()
        print(f"Successfully loaded policy from {checkpoint_path}")
    except FileNotFoundError:
        print(f"Error: Could not find checkpoint at {checkpoint_path}. Train the model first.")
        sys.exit(1)

    # Initialize environment
    env = MultiAgentGridEnv(grid_size=grid_size, num_agents=num_agents, num_targets=num_targets, num_obstacles=num_obstacles)
    obs_dict = env.reset()
    
    # Tracking metrics
    W = H = grid_size
    visit_counts = np.zeros((H, W), dtype=int)
    covered = np.zeros((H, W), dtype=bool)
    
    global_step = 0
    initial_targets = env.targets.copy()
    
    # Mark initial positions as covered/visited
    for pos in env.agents.values():
        visit_counts[pos[1], pos[0]] += 1
        mark_visible(covered, pos[0], pos[1], r=1)

    # ==========================================
    # Matplotlib Figure Setup (1x2 Layout)
    # ==========================================
    fig = plt.figure(figsize=(12.0, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.12)
    ax_world = fig.add_subplot(gs[0, 0])
    ax_shared = fig.add_subplot(gs[0, 1])
    
    # --- World Subplot (Left) ---
    world_img = ax_world.imshow(coverage_to_image(covered).transpose(1, 0, 2), origin='lower', extent=[0, W, 0, H])
    ax_world.set_title("World — Decentralized Execution (IDQN)")
    ax_world.set_xlim(0, W)
    ax_world.set_ylim(0, H)
    ax_world.set_aspect('equal', adjustable='box')
    ax_world.set_xticks(np.arange(0, W + 1, 10))
    ax_world.set_yticks(np.arange(0, H + 1, 10))
    ax_world.set_xticks(np.arange(0, W + 1, 1), minor=True)
    ax_world.set_yticks(np.arange(0, H + 1, 1), minor=True)
    ax_world.grid(which='major', color='k', alpha=0.15, linewidth=0.5)
    ax_world.grid(which='minor', color='k', alpha=0.05, linewidth=0.2)
    
    # Obstacles
    obstacles_arr = np.array(list(env.obstacles))
    if len(obstacles_arr) > 0:
        ax_world.scatter(obstacles_arr[:, 0] + 0.5, obstacles_arr[:, 1] + 0.5, s=150, marker='s', c='black', label='Obstacles')
        ax_shared.scatter(obstacles_arr[:, 0] + 0.5, obstacles_arr[:, 1] + 0.5, s=150, marker='s', c='black')

    # Robots & Labels
    robot_colors = ['k' for _ in range(num_agents)]
    initial_positions = np.array([[pos[0] + 0.5, pos[1] + 0.5] for pos in env.agents.values()])
    robot_scatter = ax_world.scatter(initial_positions[:, 0], initial_positions[:, 1], s=40, marker='o', c=robot_colors, zorder=4)
    
    robot_labels = []
    for i, pos in env.agents.items():
        txt = ax_world.text(pos[0] + 0.6, pos[1] + 0.6, f"R{i}", fontsize=7, color='k', zorder=6)
        robot_labels.append(txt)
        
    # Visited paths
    visited_scatters = []
    robot_path_history = {i: [pos] for i, pos in env.agents.items()}
    for i in range(num_agents):
        vis_sc = ax_world.scatter([], [], s=10, marker='s', facecolors='none', edgecolors='0.2', alpha=0.55, linewidths=0.6, zorder=2)
        visited_scatters.append(vis_sc)
        
    # World Targets
    if initial_targets:
        tx_world, ty_world = zip(*initial_targets)
    else:
        tx_world, ty_world = [], []
    world_targets_plot = ax_world.scatter([x + 0.5 for x in tx_world], [y + 0.5 for y in ty_world], s=20, marker='x', c='r', alpha=0.9, zorder=4)

    # --- Shared Map Subplot (Right) ---
    shared_img = ax_shared.imshow(coverage_to_image(covered).transpose(1, 0, 2), origin='lower', extent=[0, W, 0, H])
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
    
    disc_plot = ax_shared.scatter([], [], s=25, marker='o', facecolors='none', edgecolors='g', linewidths=1.5, label='Discovered')
    und_plot = ax_shared.scatter([x + 0.5 for x in tx_world], [y + 0.5 for y in ty_world], s=18, marker='x', c='r', label='Undiscovered')
    ax_shared.legend(loc='upper right', fontsize=8, frameon=False)

    # Dictionary to keep state inside the animation loop
    state = {
        'obs': obs_dict,
        'step': 0,
        'done': False
    }

    def update(frame):
        if state['done']:
            return world_img, shared_img, robot_scatter, disc_plot, und_plot, *robot_labels, *visited_scatters

        current_obs = state['obs']
        action_dict = {}
        
        # Policy Inference
        for agent_id, obs in current_obs.items():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = policy_net(obs_tensor)
            action_dict[agent_id] = q_values.argmax().item()
            
        # Step environment
        next_obs, rewards, done, _ = env.step(action_dict)
        state['step'] += 1
        state['obs'] = next_obs
        state['done'] = done
        
        # Update trackers & coverage
        for agent_id, pos in env.agents.items():
            x, y = pos
            visit_counts[y, x] += 1
            mark_visible(covered, x, y, r=1)
            robot_path_history[agent_id].append((x, y))
            
        current_undiscovered = env.targets
        current_found = initial_targets - current_undiscovered
        
        # --- Update Visuals ---
        img_data = coverage_to_image(covered).transpose(1, 0, 2)
        world_img.set_data(img_data)
        shared_img.set_data(img_data)
        
        # Robot Positions
        positions = np.array([[pos[0] + 0.5, pos[1] + 0.5] for pos in env.agents.values()])
        robot_scatter.set_offsets(positions)
        
        for i, pos in env.agents.items():
            robot_labels[i].set_position((pos[0] + 0.6, pos[1] + 0.6))
            
            # Historical paths
            hist_pts = np.array([[px + 0.5, py + 0.5] for (px, py) in robot_path_history[i]])
            visited_scatters[i].set_offsets(hist_pts)
            
        # Targets
        if current_undiscovered:
            ux, uy = zip(*current_undiscovered)
            und_plot.set_offsets(np.c_[[x + 0.5 for x in ux], [y + 0.5 for y in uy]])
            world_targets_plot.set_offsets(np.c_[[x + 0.5 for x in ux], [y + 0.5 for y in uy]])
        else:
            und_plot.set_offsets(np.empty((0, 2)))
            world_targets_plot.set_offsets(np.empty((0, 2)))
            
        if current_found:
            fx, fy = zip(*current_found)
            disc_plot.set_offsets(np.c_[[x + 0.5 for x in fx], [y + 0.5 for y in fy]])
            
        # Update Titles
        ax_world.set_title(
            f"World — Decentralized Execution (IDQN)\n"
            f"Covered: {covered.sum()} / {W*H} cells, Found targets: {len(current_found)} / {len(initial_targets)}"
        )

        if done:
            print(f"\nSimulation Terminated.")
            print(f"Total Mission Time (Makespan): {state['step']} steps to discover all targets.")
            anim.event_source.stop()
            
            # Reformat robot objects to match the requested plot_visit_counts signature
            dummy_robots = [type('Robot', (object,), {'x': pos[0], 'y': pos[1]})() for pos in env.agents.values()]
            
            # Trigger final static plot
            plot_visit_counts(
                visit_counts=visit_counts,
                robots=dummy_robots,
                targets=initial_targets,
                found_targets=current_found,
                output_path=Path('output_frames/idqn_heatmap.png')
            )

        return world_img, shared_img, robot_scatter, disc_plot, und_plot, *robot_labels, *visited_scatters

    anim = animation.FuncAnimation(fig, update, frames=2000, interval=100, blit=False)
    plt.show()



if __name__ == "__main__":
    visualize_trained_policy()