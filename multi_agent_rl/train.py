import torch
import os
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from environment import MultiAgentGridEnv
from model import DQNPolicy, ReplayBuffer

def train():
    # Hyperparameters
    num_episodes = 2000
    batch_size = 64
    gamma = 0.99
    lr = 1e-4
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.995
    target_update_freq = 10
    episode_length = 500
    
    checkpoint_dir = "checkpoints"
    checkpoint_path = f"{checkpoint_dir}/best_policy.pth"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_reward = -float('inf')

    env = MultiAgentGridEnv(grid_size=100, num_agents=10, num_targets=20, num_obstacles=0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQNPolicy().to(device)

    # --- Check for and load existing weights ---
    if os.path.exists(checkpoint_path):
        print(f"Loading existing policy from {checkpoint_path}...")
        # weights_only=True silences the pickle warning
        policy_net.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    else:
        print("No prior checkpoint found. Starting training with random weights.")

    target_net = DQNPolicy().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory = ReplayBuffer()
    
    epsilon = epsilon_start

    for episode in range(num_episodes):
        obs_dict = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        
        while not done and step_count < episode_length: # Max steps per episode
            action_dict = {}
            
            # Select actions for all agents
            for agent_id, obs in obs_dict.items():
                if np.random.rand() < epsilon:
                    action_dict[agent_id] = np.random.randint(4)
                else:
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    with torch.no_grad():
                        q_values = policy_net(obs_tensor)
                    action_dict[agent_id] = q_values.argmax().item()
                    
            next_obs_dict, rewards, done, _ = env.step(action_dict)
            
            # Store individual experiences in the shared buffer
            for i in range(env.num_agents):
                memory.push(obs_dict[i], action_dict[i], rewards[i], next_obs_dict[i], done)
                total_reward += rewards[i]
                
            obs_dict = next_obs_dict
            step_count += 1
            
            # Training Step
            if len(memory) > batch_size:
                states, actions, batch_rewards, next_states, dones = memory.sample(batch_size)
                
                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                batch_rewards = torch.FloatTensor(batch_rewards).unsqueeze(1).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
                
                # Q-Learning update
                curr_Q = policy_net(states).gather(1, actions)
                with torch.no_grad():
                    max_next_Q = target_net(next_states).max(1)[0].unsqueeze(1)
                    target_Q = batch_rewards + (gamma * max_next_Q * (1 - dones))
                    
                loss = F.mse_loss(curr_Q, target_Q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        # Epsilon decay
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Target Network Update
        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        if episode % 50 == 0:
            print(f"Episode {episode} | Steps: {step_count} | Total Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")

            # Save if this is the best performing policy so far
            if total_reward > best_reward:
                best_reward = total_reward
                torch.save(policy_net.state_dict(), "checkpoints/best_policy.pth")
                print(f"--> New best model saved! (Reward: {best_reward:.2f})")
    
    # Save the final state at the very end of the loop
    torch.save(policy_net.state_dict(), "checkpoints/final_policy.pth")
    print("Training complete. Models saved to /checkpoints.")

if __name__ == "__main__":
    train()