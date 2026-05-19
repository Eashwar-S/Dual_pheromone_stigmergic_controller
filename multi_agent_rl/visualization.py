import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CURRENT_DIR))

from common.simulation import run_animation
from common.utilities import mark_visible
from common.visualization import coverage_to_image, plot_visit_counts, save_targets_over_time_plot
from environment import MultiAgentGridEnv
from model import DQNPolicy


def _robot_points(agents):
    return np.array([[pos[0] + 0.5, pos[1] + 0.5] for pos in agents.values()])


def visualize_trained_policy(checkpoint_path=None,
                             grid_size=100, num_agents=10, num_targets=5,
                             num_obstacles=0, seed=7, robot_radius=1,
                             steps_per_frame=10, interval_ms=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQNPolicy(input_channels=4, fov_size=3, action_space=4).to(device)

    checkpoint = Path(checkpoint_path) if checkpoint_path else CURRENT_DIR / "checkpoints" / "best_policy.pth"
    if not checkpoint.exists():
        print(f"Error: Could not find checkpoint at {checkpoint}. Train the model first.")
        sys.exit(1)

    policy_net.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    policy_net.eval()
    print(f"Loaded policy from {checkpoint}")

    env = MultiAgentGridEnv(
        grid_size=grid_size,
        num_agents=num_agents,
        num_targets=num_targets,
        num_obstacles=num_obstacles,
        seed=seed,
    )
    obs_dict = env.reset(seed=seed)

    W = H = grid_size
    output_dir = CURRENT_DIR / "output_frames" / "multi_agent_rl"
    targets_found_over_time = []
    initial_targets = set(env.targets)
    covered = np.zeros((H, W), dtype=bool)
    visit_counts = np.zeros((H, W), dtype=int)
    robot_path_history = {i: [pos] for i, pos in env.agents.items()}

    for pos in env.agents.values():
        mark_visible(covered, pos[0], pos[1], robot_radius)
        mark_visible(visit_counts, pos[0], pos[1], robot_radius)

    fig = plt.figure(figsize=(12.0, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.12)
    ax_world = fig.add_subplot(gs[0, 0])
    ax_shared = fig.add_subplot(gs[0, 1])

    world_img = ax_world.imshow(coverage_to_image(covered), origin="lower", extent=[0, W, 0, H], vmin=0.0, vmax=1.0)
    shared_img = ax_shared.imshow(coverage_to_image(covered), origin="lower", extent=[0, W, 0, H], vmin=0.0, vmax=1.0)

    for ax in (ax_world, ax_shared):
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks(np.arange(0, W + 1, 10))
        ax.set_yticks(np.arange(0, H + 1, 10))
        ax.set_xticks(np.arange(0, W + 1, 1), minor=True)
        ax.set_yticks(np.arange(0, H + 1, 1), minor=True)
        ax.grid(which="major", color="k", alpha=0.15, linewidth=0.5)
        ax.grid(which="minor", color="k", alpha=0.05, linewidth=0.2)

    ax_world.set_title("World - Decentralized Execution (Multi-agent RL)")
    ax_shared.set_title("Shared Map (Global Knowledge)")

    robot_scatter = ax_world.scatter(
        _robot_points(env.agents)[:, 0],
        _robot_points(env.agents)[:, 1],
        s=40,
        marker="o",
        c="k",
        zorder=4,
    )
    robot_labels = [
        ax_world.text(pos[0] + 0.6, pos[1] + 0.6, f"R{i}", fontsize=7, color="k", zorder=6)
        for i, pos in env.agents.items()
    ]
    visited_scatters = [
        ax_world.scatter([], [], s=10, marker="s", facecolors="none",
                         edgecolors="0.2", alpha=0.55, linewidths=0.6, zorder=2)
        for _ in range(num_agents)
    ]

    if initial_targets:
        tx, ty = zip(*initial_targets)
    else:
        tx, ty = [], []
    world_targets_plot = ax_world.scatter(
        [x + 0.5 for x in tx], [y + 0.5 for y in ty],
        s=20, marker="x", c="r", alpha=0.9, zorder=4,
    )
    disc_plot = ax_shared.scatter([], [], s=25, marker="o", facecolors="none",
                                  edgecolors="g", linewidths=1.5, label="Discovered")
    und_plot = ax_shared.scatter([x + 0.5 for x in tx], [y + 0.5 for y in ty],
                                 s=18, marker="x", c="r", label="Undiscovered")
    ax_shared.legend(loc="upper right", fontsize=8, frameon=False)

    state = {"obs": obs_dict, "step": 0, "done": False, "plot_saved": False}

    def sim_step():
        action_dict = {}
        for agent_id, obs in state["obs"].items():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action_dict[agent_id] = int(policy_net(obs_tensor).argmax().item())

        next_obs, _, done, _ = env.step(action_dict)
        state["obs"] = next_obs
        state["step"] += 1

        for agent_id, pos in env.agents.items():
            x, y = pos
            mark_visible(covered, x, y, robot_radius)
            mark_visible(visit_counts, x, y, robot_radius)
            robot_path_history[agent_id].append((x, y))

        current_undiscovered = set(env.targets)
        current_found = initial_targets - current_undiscovered
        targets_found_over_time.append(len(current_found))
        state["done"] = done
        return current_undiscovered, current_found

    def update(_frame):
        if state["done"]:
            return world_img, shared_img, robot_scatter, disc_plot, und_plot, *robot_labels, *visited_scatters

        current_undiscovered = set(env.targets)
        current_found = initial_targets - current_undiscovered
        for _ in range(steps_per_frame):
            current_undiscovered, current_found = sim_step()
            if state["done"]:
                break

        img = coverage_to_image(covered)
        world_img.set_data(img)
        shared_img.set_data(img)
        robot_scatter.set_offsets(_robot_points(env.agents))

        for i, pos in env.agents.items():
            robot_labels[i].set_position((pos[0] + 0.6, pos[1] + 0.6))
            hist = np.array([[x + 0.5, y + 0.5] for x, y in robot_path_history[i]])
            visited_scatters[i].set_offsets(hist)

        if current_undiscovered:
            ux, uy = zip(*current_undiscovered)
            points = np.c_[[x + 0.5 for x in ux], [y + 0.5 for y in uy]]
            und_plot.set_offsets(points)
            world_targets_plot.set_offsets(points)
        else:
            und_plot.set_offsets(np.empty((0, 2)))
            world_targets_plot.set_offsets(np.empty((0, 2)))

        if current_found:
            fx, fy = zip(*current_found)
            disc_plot.set_offsets(np.c_[[x + 0.5 for x in fx], [y + 0.5 for y in fy]])

        ax_world.set_title(
            "World - Decentralized Execution (Multi-agent RL)\n"
            f"Covered: {covered.sum()} / {W * H} cells, Found targets: {len(current_found)} / {len(initial_targets)}"
        )

        if state["done"] and not state["plot_saved"]:
            output_dir.mkdir(parents=True, exist_ok=True)
            save_targets_over_time_plot(output_dir / "targets_over_time.png", targets_found_over_time)
            np.save(output_dir / "multi_agent_rl_without_failure.npy", np.array(targets_found_over_time))
            state["plot_saved"] = True

        return world_img, shared_img, robot_scatter, disc_plot, und_plot, *robot_labels, *visited_scatters

    anim = run_animation(fig, update, frames=2000000, interval_ms=interval_ms, blit=False)
    plt.show()
    robots = [SimpleNamespace(x=pos[0], y=pos[1]) for pos in env.agents.values()]
    plot_visit_counts(
        visit_counts=visit_counts,
        robots=robots,
        targets=initial_targets,
        found_targets=initial_targets - set(env.targets),
        output_path=output_dir / "final_visit_counts.png",
    )


if __name__ == "__main__":
    STEPS_PER_FRAME = 30
    INTERVAL_MS = 50
    visualize_trained_policy(steps_per_frame=STEPS_PER_FRAME, interval_ms=INTERVAL_MS)
