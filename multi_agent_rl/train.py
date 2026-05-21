import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CURRENT_DIR))

import config_experiments
from environment import MultiAgentGridEnv
from model import DQNPolicy, ReplayBuffer


def _load_dotenv():
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(CURRENT_DIR / ".env", override=True)


def _init_wandb(config):
    _load_dotenv()
    mode = os.getenv("WANDB_MODE", "disabled").lower()
    if mode == "disabled":
        return None

    try:
        import wandb
    except ImportError:
        print("wandb is not installed. Continuing without W&B logging.")
        return None

    api_key = os.getenv("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)

    return wandb.init(
        project=os.getenv("WANDB_PROJECT", "stigmergy-marl"),
        entity=os.getenv("WANDB_ENTITY") or None,
        mode=mode,
        config=config,
    )


def _set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_env(config):
    grid_size, n_robots, n_targets, _ = config
    return MultiAgentGridEnv(
        grid_size=grid_size,
        num_agents=n_robots,
        num_targets=n_targets,
        num_obstacles=0,
    )


def _select_actions(policy_net, obs_dict, epsilon, rng, device):
    action_dict = {}
    for agent_id, obs in obs_dict.items():
        if rng.random() < epsilon:
            action_dict[agent_id] = int(rng.integers(4))
        else:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = policy_net(obs_tensor)
            action_dict[agent_id] = int(q_values.argmax().item())
    return action_dict


def _optimize(policy_net, target_net, optimizer, memory, batch_size, gamma, device):
    if len(memory) <= batch_size:
        return None

    states, actions, batch_rewards, next_states, dones = memory.sample(batch_size)
    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    batch_rewards = torch.FloatTensor(batch_rewards).unsqueeze(1).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

    curr_q = policy_net(states).gather(1, actions)
    with torch.no_grad():
        max_next_q = target_net(next_states).max(1)[0].unsqueeze(1)
        target_q = batch_rewards + (gamma * max_next_q * (1 - dones))

    loss = F.mse_loss(curr_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
    optimizer.step()
    return float(loss.item())


def _run_episode(policy_net, env, seed, epsilon, episode_length, device, rng, memory=None,
                 target_net=None, optimizer=None, batch_size=64, gamma=0.99):
    obs_dict = env.reset(seed=seed)
    total_reward = 0.0
    losses = []
    done = False
    step_count = 0
    last_info = env.get_metrics()

    while not done and step_count < episode_length:
        action_dict = _select_actions(policy_net, obs_dict, epsilon, rng, device)
        next_obs_dict, rewards, done, info = env.step(action_dict)

        if memory is not None:
            for agent_id in obs_dict:
                memory.push(obs_dict[agent_id], action_dict[agent_id], rewards[agent_id], next_obs_dict[agent_id], done)

            loss = _optimize(policy_net, target_net, optimizer, memory, batch_size, gamma, device)
            if loss is not None:
                losses.append(loss)

        total_reward += sum(rewards.values())
        obs_dict = next_obs_dict
        step_count += 1
        last_info = info

    return {
        "reward": total_reward,
        "steps": step_count,
        "loss": float(np.mean(losses)) if losses else 0.0,
        "coverage_percent": last_info["coverage_percent"],
        "targets_found": last_info["targets_found"],
        "target_fraction": last_info["target_fraction"],
        "collisions": last_info["total_collisions"],
        "revisits": last_info["total_revisits"],
        "new_global_cells": last_info["total_new_global_cells"],
    }


def evaluate(policy_net, configs, seeds, episode_length, device):
    policy_net.eval()
    metrics = []
    by_grid = {}
    eval_rng = np.random.default_rng(0)

    with torch.no_grad():
        for config in configs:
            grid_size = config[0]
            grid_metrics = []
            for seed in seeds:
                env = _make_env(config)
                result = _run_episode(
                    policy_net=policy_net,
                    env=env,
                    seed=seed,
                    epsilon=0.0,
                    episode_length=episode_length,
                    device=device,
                    rng=eval_rng,
                )
                grid_metrics.append(result)
                metrics.append(result)

            by_grid[grid_size] = {
                "mean_found": float(np.mean([m["target_fraction"] for m in grid_metrics])),
                "percent_coverage": float(np.mean([m["coverage_percent"] for m in grid_metrics])),
                "mean_reward": float(np.mean([m["reward"] for m in grid_metrics])),
            }

    policy_net.train()
    overall_score = float(np.mean([
        0.7 * m["target_fraction"] + 0.3 * (m["coverage_percent"] / 100.0)
        for m in metrics
    ])) if metrics else 0.0

    return {
        "overall_score": overall_score,
        "mean_found": float(np.mean([m["target_fraction"] for m in metrics])) if metrics else 0.0,
        "percent_coverage": float(np.mean([m["coverage_percent"] for m in metrics])) if metrics else 0.0,
        "by_grid": by_grid,
    }


def _log_wandb(run, values, step=None):
    if run is not None:
        run.log(values, step=step)


def _save_artifact(run, path, name, artifact_type="model"):
    if run is None:
        return
    import wandb
    artifact = wandb.Artifact(name, type=artifact_type)
    artifact.add_file(str(path))
    run.log_artifact(artifact)


def train(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--target-update-freq", type=int, default=10)
    parser.add_argument("--episode-length", type=int, default=1000)
    parser.add_argument("--eval-freq", type=int, default=50)
    parser.add_argument("--eval-episodes-per-config", type=int, default=2)
    parser.add_argument("--seed", type=int, default=config_experiments.BASE_SEED)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--resume", action="store_true")
    parsed = parser.parse_args(args)

    _set_global_seeds(parsed.seed)

    configs = config_experiments.get_experiment_configs()
    train_rng = np.random.default_rng(parsed.seed)
    eval_seeds = [parsed.seed + 10000 + i for i in range(parsed.eval_episodes_per_config)]

    checkpoint_dir = Path(parsed.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    best_checkpoint = checkpoint_dir / "best_policy.pth"
    final_checkpoint = checkpoint_dir / "final_policy.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQNPolicy().to(device)
    if parsed.resume and best_checkpoint.exists():
        print(f"Loading existing policy from {best_checkpoint}...")
        policy_net.load_state_dict(torch.load(best_checkpoint, map_location=device, weights_only=True))
    else:
        print("Starting training with random weights.")

    target_net = DQNPolicy().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=parsed.lr)
    memory = ReplayBuffer()
    epsilon = parsed.epsilon_start
    best_score = -float("inf")

    run_config = vars(parsed).copy()
    run_config.update({
        "num_configs": len(configs),
        "device": str(device),
        "grid_sizes": [c[0] for c in configs],
        "robot_counts": [c[1] for c in configs],
        "target_counts": [c[2] for c in configs],
    })
    wandb_run = _init_wandb(run_config)

    for episode in range(1, parsed.episodes + 1):
        config = configs[int(train_rng.integers(len(configs)))]
        env = _make_env(config)
        episode_seed = parsed.seed + episode

        result = _run_episode(
            policy_net=policy_net,
            env=env,
            seed=episode_seed,
            epsilon=epsilon,
            episode_length=parsed.episode_length,
            device=device,
            rng=train_rng,
            memory=memory,
            target_net=target_net,
            optimizer=optimizer,
            batch_size=parsed.batch_size,
            gamma=parsed.gamma,
        )

        epsilon = max(parsed.epsilon_end, epsilon * parsed.epsilon_decay)
        if episode % parsed.target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        log_values = {
            "train/reward": result["reward"],
            "train/steps": result["steps"],
            "train/loss": result["loss"],
            "train/epsilon": epsilon,
            "train/targets_found": result["targets_found"],
            "train/target_fraction": result["target_fraction"],
            "train/coverage_percent": result["coverage_percent"],
            "train/collisions": result["collisions"],
            "train/revisits": result["revisits"],
            "train/new_global_cells": result["new_global_cells"],
            "train/grid_size": config[0],
            "train/n_robots": config[1],
            "train/n_targets": config[2],
        }
        _log_wandb(wandb_run, log_values, step=episode)

        if episode % parsed.eval_freq == 0 or episode == 1:
            eval_result = evaluate(policy_net, configs, eval_seeds, parsed.episode_length, device)
            eval_log = {
                "eval/overall_score": eval_result["overall_score"],
                "eval/mean_found": eval_result["mean_found"],
                "eval/percent_coverage": eval_result["percent_coverage"],
            }
            for grid_size, grid_metrics in eval_result["by_grid"].items():
                eval_log[f"eval/grid_{grid_size}/mean_found"] = grid_metrics["mean_found"]
                eval_log[f"eval/grid_{grid_size}/percent_coverage"] = grid_metrics["percent_coverage"]
                eval_log[f"eval/grid_{grid_size}/mean_reward"] = grid_metrics["mean_reward"]

            _log_wandb(wandb_run, eval_log, step=episode)
            print(
                f"Episode {episode} | grid={config[0]} robots={config[1]} targets={config[2]} "
                f"| reward={result['reward']:.2f} | coverage={result['coverage_percent']:.2f}% "
                f"| eval_score={eval_result['overall_score']:.4f} | epsilon={epsilon:.3f}"
            )

            if eval_result["overall_score"] > best_score:
                best_score = eval_result["overall_score"]
                torch.save(policy_net.state_dict(), best_checkpoint)
                _save_artifact(wandb_run, best_checkpoint, "best_policy")
                print(f"--> New best model saved! (Eval score: {best_score:.4f})")

    torch.save(policy_net.state_dict(), final_checkpoint)
    _save_artifact(wandb_run, final_checkpoint, "final_policy")
    if wandb_run is not None:
        wandb_run.finish()

    print(f"Training complete. Models saved to {checkpoint_dir}.")


if __name__ == "__main__":
    train()
