import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import random
import sys
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent                        
sys.path.insert(0, str(PROJECT_ROOT))
from stigmergy_common.pheromone import deposit_uniform


@dataclass
class Robot:
    """Robot using pure lexicographical stigmergy with a Stateful Self-Avoiding Random Walk."""
    id: int
    x: int
    y: int
    local_covered: np.ndarray
    failed: bool = False
    
    _last_move: Optional[Tuple[int, int]] = None
    _position_visits: Dict[Tuple[int, int], int] = field(default_factory=dict, repr=False)
    _in_escape_mode: bool = False
    _escape_visited: Set[Tuple[int, int]] = field(default_factory=set, repr=False)

    def is_clear(self, nx: int, ny: int, all_robots: List['Robot'], radius: int = 1) -> bool:
        """Checks if moving to (nx, ny) violates the collision radius of any OTHER robot."""
        for r in all_robots:
            if r.id == self.id:
                continue
            if max(abs(r.x - nx), abs(r.y - ny)) <= radius:
                return False
        return True

    def choose_move_search(self, pher_rep: np.ndarray, all_robots: List['Robot'],
                           robot_radius: int, collision_radius: int = 1) -> Tuple[int, int]:
        H, W = pher_rep.shape

        if not self._position_visits:
            self._position_visits[(self.x, self.y)] = 1

        actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        random.shuffle(actions)
        candidates = []

        for dx, dy in actions:
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < W and 0 <= ny < H and self.is_clear(nx, ny, all_robots, collision_radius):
                
                y_min, y_max = max(0, ny - robot_radius), min(H, ny + robot_radius + 1)
                x_min, x_max = max(0, nx - robot_radius), min(W, nx + robot_radius + 1)
                footprint_pher = np.sum(pher_rep[y_min:y_max, x_min:x_max])
                
                visits = self._position_visits.get((nx, ny), 0)
                momentum_score = -1 if (self._last_move == (dx, dy)) else 0

                candidates.append({
                    'move': (nx, ny),
                    'action': (dx, dy),
                    'pher': footprint_pher,
                    'visits': visits,
                    'momentum': momentum_score
                })

        if not candidates:
            return (self.x, self.y)

        # If every available move has already been visited, we are stuck in a local minimum.
        all_visited = all(c['visits'] > 0 for c in candidates)
        
        if all_visited and not self._in_escape_mode:
            self._in_escape_mode = True
            self._escape_visited.clear()
            self._escape_visited.add((self.x, self.y))

        # Execute Intelligent Random Walk (Escape Mode)
        if self._in_escape_mode:
            # Filter out moves we have already taken *during this specific escape sequence*
            escape_candidates = [c for c in candidates if c['move'] not in self._escape_visited]
            
            if not escape_candidates:
                # Dead End in Escape Mode (e.g., boxed in a corner). 
                # Clear the short-term memory to allow backtracking, but stay in escape mode.
                self._escape_visited.clear()
                self._escape_visited.add((self.x, self.y))
                escape_candidates = candidates

            # Pure random choice among valid escape routes
            chosen = random.choice(escape_candidates)
            
            nx, ny = chosen['move']
            
            # Extract the exact local pheromone of the chosen cell
            local_pher = pher_rep[ny, nx]
            
            swarm_unvisited = (local_pher == 0.0)
            
            self_unvisited = (self._position_visits.get((nx, ny), 0) == 0)

            if swarm_unvisited or self_unvisited:
                self._in_escape_mode = False
                self._escape_visited.clear()
            else:
                self._escape_visited.add(chosen['move'])
                
            return chosen['move']

        # Lexicographical sorting
        candidates.sort(key=lambda c: (round(c['pher'], 5), c['visits'], c['momentum']))
        best = candidates[0]
        
        return best['move']

    def step(self, pher_rep: np.ndarray, all_robots: List['Robot'], 
             robot_radius: int, collision_radius: int = 1):
        """Execute one step: choose move and update position."""
        if self.failed:
            return
        
        old_x, old_y = self.x, self.y
        nx, ny = self.choose_move_search(pher_rep, all_robots, robot_radius, collision_radius)
        
        self.x, self.y = nx, ny
        self._last_move = (nx - old_x, ny - old_y)
        self._position_visits[(nx, ny)] = self._position_visits.get((nx, ny), 0) + 1

    def deposit_pheromone(self, pher: np.ndarray, amount: float, r: int = 5):
        """Deposit repulsive pheromone in neighborhood."""
        deposit_uniform(pher, self.x, self.y, amount, r)
