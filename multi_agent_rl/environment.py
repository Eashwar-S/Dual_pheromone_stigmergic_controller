import numpy as np

class MultiAgentGridEnv:
    def __init__(self, grid_size=10, num_agents=3, num_targets=5, num_obstacles=5,
                 step_penalty=-0.01, collision_penalty=-1.0, revisit_penalty=-0.05,
                 new_local_cell_reward=0.02, new_global_cell_reward=0.05,
                 target_reward=10.0, team_target_reward=1.0):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.num_targets = num_targets
        self.num_obstacles = num_obstacles
        self.step_penalty = step_penalty
        self.collision_penalty = collision_penalty
        self.revisit_penalty = revisit_penalty
        self.new_local_cell_reward = new_local_cell_reward
        self.new_global_cell_reward = new_global_cell_reward
        self.target_reward = target_reward
        self.team_target_reward = team_target_reward
        
        # Actions: 0: Up, 1: Down, 2: Left, 3: Right
        self.action_space = 4
        self.reset()

    def reset(self, seed=None, rng=None):
        if rng is None:
            rng = np.random.default_rng(seed)

        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.visited_memory = {i: np.zeros((self.grid_size, self.grid_size)) for i in range(self.num_agents)}
        self.global_visited = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.agents = {}
        self.targets = set()
        self.initial_num_targets = self.num_targets
        self.obstacles = set()
        self.steps = 0
        self.total_collisions = 0
        self.total_revisits = 0
        self.total_new_global_cells = 0
        self.total_new_local_cells = 0
        self.total_targets_found = 0
        
        # Place Obstacles
        while len(self.obstacles) < self.num_obstacles:
            pos = (int(rng.integers(self.grid_size)), int(rng.integers(self.grid_size)))
            self.obstacles.add(pos)
            
        # Place Targets
        while len(self.targets) < self.num_targets:
            pos = (int(rng.integers(self.grid_size)), int(rng.integers(self.grid_size)))
            if pos not in self.obstacles:
                self.targets.add(pos)
                
        # Place Agents
        for i in range(self.num_agents):
            while True:
                pos = (int(rng.integers(self.grid_size)), int(rng.integers(self.grid_size)))
                if pos not in self.obstacles and pos not in self.agents.values():
                    self.agents[i] = pos
                    self.visited_memory[i][pos] = 1 # Initialize memory
                    if not self.global_visited[pos]:
                        self.global_visited[pos] = True
                        self.total_new_global_cells += 1
                    break
                    
        return self._get_all_observations()

    def _get_local_obs(self, agent_id):
        # Returns a 4x3x3 tensor: [Obstacles/Bounds, Other Agents, Targets, Memory]
        x, y = self.agents[agent_id]
        obs = np.zeros((4, 3, 3), dtype=np.float32)
        
        # Von Neumann neighborhood: Center (0,0) + 4 cardinal directions
        valid_offsets = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
        
        for dx, dy in valid_offsets:
            nx, ny = x + dx, y + dy
            
            # Map the (-1 to 1) offset to the (0 to 2) tensor index
            obs_x, obs_y = dx + 1, dy + 1
            
            # Check boundaries
            if nx < 0 or nx >= self.grid_size or ny < 0 or ny >= self.grid_size:
                obs[0, obs_x, obs_y] = 1.0 # Obstacle channel
                continue
                
            pos = (nx, ny)
            if pos in self.obstacles:
                obs[0, obs_x, obs_y] = 1.0
            if pos in self.targets:
                obs[2, obs_x, obs_y] = 1.0
            
            # Check for other agents (collision avoidance)
            if any(p == pos for i, p in self.agents.items() if i != agent_id):
                obs[1, obs_x, obs_y] = 1.0
                
            # Local memory scalar (capped at 5 and normalized to 0.0 - 1.0)
            obs[3, obs_x, obs_y] = min(self.visited_memory[agent_id][nx, ny], 5.0) / 5.0
                
        return obs

    def _get_all_observations(self):
        return {i: self._get_local_obs(i) for i in range(self.num_agents)}

    def step(self, action_dict):
        self.steps += 1
        rewards = {i: self.step_penalty for i in range(self.num_agents)}
        info = {
            "collisions": 0,
            "revisits": 0,
            "new_global_cells": 0,
            "new_local_cells": 0,
            "targets_found": 0,
        }
        
        # Calculate intended moves
        intended_positions = {}
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        
        for agent_id, action in action_dict.items():
            x, y = self.agents[agent_id]
            dx, dy = moves[action]
            intended_positions[agent_id] = (x + dx, y + dy)

        # Resolve collisions
        final_positions = {}
        for agent_id, pos in intended_positions.items():
            x, y = pos
            # Boundary or static obstacle collision
            if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size or pos in self.obstacles:
                final_positions[agent_id] = self.agents[agent_id] # Move denied
                rewards[agent_id] += self.collision_penalty
                info["collisions"] += 1
                continue
                
            # Agent-Agent collision (two agents moving to same spot)
            if list(intended_positions.values()).count(pos) > 1:
                final_positions[agent_id] = self.agents[agent_id] # Move denied
                rewards[agent_id] += self.collision_penalty
                info["collisions"] += 1
                continue
                
            # Agent-Agent swap collision
            swapped = False
            for other_id, other_intended in intended_positions.items():
                if agent_id != other_id and pos == self.agents[other_id] and other_intended == self.agents[agent_id]:
                    final_positions[agent_id] = self.agents[agent_id]
                    rewards[agent_id] += self.collision_penalty
                    info["collisions"] += 1
                    swapped = True
                    break
            if swapped: continue
            
            # Valid move
            final_positions[agent_id] = pos

        # Update state and calculate exploration/target rewards
        for agent_id, pos in final_positions.items():
            self.agents[agent_id] = pos
            was_local_visited = self.visited_memory[agent_id][pos] > 0
            was_global_visited = self.global_visited[pos]
            self.visited_memory[agent_id][pos] += 1
            
            if was_local_visited:
                rewards[agent_id] += self.revisit_penalty
                info["revisits"] += 1
            else:
                rewards[agent_id] += self.new_local_cell_reward
                info["new_local_cells"] += 1

            if not was_global_visited:
                self.global_visited[pos] = True
                rewards[agent_id] += self.new_global_cell_reward
                info["new_global_cells"] += 1
                
            # Reward finding target
            if pos in self.targets:
                self.targets.remove(pos)
                rewards[agent_id] += self.target_reward
                for other_id in rewards:
                    if other_id != agent_id:
                        rewards[other_id] += self.team_target_reward
                info["targets_found"] += 1

        done = len(self.targets) == 0
        self.total_collisions += info["collisions"]
        self.total_revisits += info["revisits"]
        self.total_new_global_cells += info["new_global_cells"]
        self.total_new_local_cells += info["new_local_cells"]
        self.total_targets_found += info["targets_found"]

        info.update(self.get_metrics())
        return self._get_all_observations(), rewards, done, info

    def get_metrics(self):
        total_cells = self.grid_size * self.grid_size
        covered_cells = int(np.sum(self.global_visited))
        targets_found = self.initial_num_targets - len(self.targets)
        return {
            "covered_cells": covered_cells,
            "coverage_percent": (covered_cells / total_cells) * 100.0,
            "targets_found": targets_found,
            "target_fraction": targets_found / self.initial_num_targets if self.initial_num_targets else 1.0,
            "total_collisions": self.total_collisions,
            "total_revisits": self.total_revisits,
            "total_new_global_cells": self.total_new_global_cells,
            "total_new_local_cells": self.total_new_local_cells,
        }
