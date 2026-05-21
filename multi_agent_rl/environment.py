import numpy as np

class MultiAgentGridEnv:
    def __init__(self, grid_size=10, num_agents=3, num_targets=5, num_obstacles=5):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.num_targets = num_targets
        self.num_obstacles = num_obstacles
        
        # Actions: 0: Up, 1: Down, 2: Left, 3: Right
        self.action_space = 4
        self.reset()

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.visited_memory = {i: np.zeros((self.grid_size, self.grid_size)) for i in range(self.num_agents)}
        self.agents = {}
        self.targets = set()
        self.obstacles = set()
        self.steps = 0
        
        # Place Obstacles
        while len(self.obstacles) < self.num_obstacles:
            pos = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
            self.obstacles.add(pos)
            
        # Place Targets
        while len(self.targets) < self.num_targets:
            pos = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
            if pos not in self.obstacles:
                self.targets.add(pos)
                
        # Place Agents
        for i in range(self.num_agents):
            while True:
                pos = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
                if pos not in self.obstacles and pos not in self.agents.values():
                    self.agents[i] = pos
                    self.visited_memory[i][pos] = 1 # Initialize memory
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
        rewards = {i: -0.1 for i in range(self.num_agents)} # Step penalty
        
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
                rewards[agent_id] -= 5.0
                continue
                
            # Agent-Agent collision (two agents moving to same spot)
            if list(intended_positions.values()).count(pos) > 1:
                final_positions[agent_id] = self.agents[agent_id] # Move denied
                rewards[agent_id] -= 5.0
                continue
                
            # Agent-Agent swap collision
            swapped = False
            for other_id, other_intended in intended_positions.items():
                if agent_id != other_id and pos == self.agents[other_id] and other_intended == self.agents[agent_id]:
                    final_positions[agent_id] = self.agents[agent_id]
                    rewards[agent_id] -= 5.0
                    swapped = True
                    break
            if swapped: continue
            
            # Valid move
            final_positions[agent_id] = pos

        # Update state and calculate exploration/target rewards
        for agent_id, pos in final_positions.items():
            self.agents[agent_id] = pos
            self.visited_memory[agent_id][pos] += 1
            
            # Penalize revisiting
            if self.visited_memory[agent_id][pos] > 1:
                rewards[agent_id] -= 0.05
                
            # Reward finding target
            if pos in self.targets:
                self.targets.remove(pos)
                # Global reward broadcast for cooperation, or local. We'll give local +10
                rewards[agent_id] += 10.0 

        done = len(self.targets) == 0
        return self._get_all_observations(), rewards, done, {}