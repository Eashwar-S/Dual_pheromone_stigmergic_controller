import numpy as np
import random
from dataclasses import dataclass, field
from typing import Tuple, Optional, Set, List, Dict, Deque
from collections import deque
from .pheromone import deposit_uniform, deposit_distance_signal


@dataclass
class Robot:
    """Multi-mode robot with SEARCH/FOLLOW/ADVERTISE modes and collision avoidance."""
    id: int
    x: int
    y: int
    local_covered: np.ndarray
    start_x: int
    start_y: int
    mode: str = "SEARCH"
    spiral_iter: Optional[object] = None
    visited_targets: Set[Tuple[int, int]] = field(default_factory=set)
    advertising_target: Optional[Tuple[int, int]] = None
    follow_history: Deque[Tuple[int, int]] = field(default_factory=lambda: deque(maxlen=6))
    current_terms: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    failed: bool = False

    def is_clear(self, nx: int, ny: int, all_robots: List['Robot'], radius: int = 1) -> bool:
        """Check if (nx, ny) is clear of other robots within collision radius."""
        for r in all_robots:
            if r.id == self.id:
                continue
            if max(abs(r.x - nx), abs(r.y - ny)) <= radius:
                return False
        return True

    def calculate_utility(self, n_new_global: int, n_new_local: int, pher_sum: float) -> Tuple[float, Tuple[float, float, float]]:
        """Calculate utility score for search mode movement and return individual terms."""
        KAPPA = 20.0
        SIGMA = 10.0
        term1 = float(n_new_global)
        term2 = float(n_new_local) / KAPPA
        term3 = (np.tanh(-pher_sum / SIGMA) + 1.0) / (KAPPA ** 2)
        return term1 + term2 + term3, (term1, term2, term3)

    def choose_move_search(self, pher_rep: np.ndarray, pher_attr: np.ndarray, 
                           all_robots: List['Robot'], collision_radius: int) -> Tuple[int, int]:
        """Choose move in SEARCH mode avoiding pheromone and preferring unexplored areas."""
        H, W = pher_rep.shape
        candidates = []
        if self.y - 1 >= 0: candidates.append((self.x, self.y - 1))
        if self.y + 1 < H:  candidates.append((self.x, self.y + 1))
        if self.x - 1 >= 0: candidates.append((self.x - 1, self.y))
        if self.x + 1 < W:  candidates.append((self.x + 1, self.y))
        
        candidates = [c for c in candidates if self.is_clear(c[0], c[1], all_robots, collision_radius)]
        random.shuffle(candidates)
        if not candidates:
            self.current_terms = (0.0, 0.0, 0.0)
            return (self.x, self.y)

        R = 5
        scored_moves = []
        for (nx, ny) in candidates:
            nb_cells = []
            for dy in range(-R, R + 1):
                for dx in range(-R, R + 1):
                    if abs(dx) + abs(dy) <= R:
                        cx, cy = nx + dx, ny + dy
                        if 0 <= cx < W and 0 <= cy < H:
                            nb_cells.append((cx, cy))
            
            new_anyone = sum(1 for (cx, cy) in nb_cells if (not self.local_covered[cy, cx]) and (pher_rep[cy, cx] < 1e-3))
            new_me = sum(1 for (cx, cy) in nb_cells if not self.local_covered[cy, cx])
            pher_sum = float(np.sum([pher_rep[cy, cx] for (cx, cy) in nb_cells]))
            
            score, terms = self.calculate_utility(new_anyone, new_me, pher_sum)
            scored_moves.append({'move': (nx, ny), 'score': score, 'u1': new_anyone, 'pher': pher_rep[ny, nx], 'terms': terms})

        best = max(scored_moves, key=lambda x: x['score'])
        self.current_terms = best['terms']
        
        if best['u1'] == 0:
            if not hasattr(self, "_escape_steps") or self._escape_steps <= 0:
                min_edge = min(m['pher'] for m in scored_moves)
                starters = [m['move'] for m in scored_moves if m['pher'] == min_edge]
                ex, ey = random.choice(starters)
                self._escape_dir = (ex - self.x, ey - self.y)
                self._escape_steps = 15
            
            dx, dy = getattr(self, "_escape_dir", (0, 0))
            tx, ty = self.x + dx, self.y + dy
            
            if 0 <= tx < W and 0 <= ty < H and self.is_clear(tx, ty, all_robots, collision_radius):
                self._escape_steps -= 1
                return (tx, ty)
            else:
                self._escape_steps = 0
                return random.choice(candidates)
        
        if hasattr(self, "_escape_steps"):
            self._escape_steps = 0
        return best['move']

    def choose_move_follow(self, pher_attr: np.ndarray, all_robots: List['Robot'], 
                           collision_radius: int) -> Tuple[int, int]:
        """Choose move in FOLLOW mode using center-of-gravity of pheromone."""
        H, W = pher_attr.shape
        R = 5
        
        vec_x, vec_y = 0.0, 0.0
        total_weight = 0.0
        found_blue = False
        
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                if abs(dx) + abs(dy) <= R:
                    cx, cy = self.x + dx, self.y + dy
                    if 0 <= cx < W and 0 <= cy < H:
                        val = pher_attr[cy, cx]
                        if val > 1e-6:
                            found_blue = True
                            weight = val * val
                            vec_x += dx * weight
                            vec_y += dy * weight
                            total_weight += weight

        if not found_blue:
            self.mode = "SEARCH"
            return (self.x, self.y)

        if total_weight > 0:
            length = np.sqrt(vec_x**2 + vec_y**2)
            if length > 0:
                vec_x /= length
                vec_y /= length
        else:
            vec_x, vec_y = 0, 0

        candidates = []
        if self.y - 1 >= 0: candidates.append((0, -1, self.x, self.y - 1))
        if self.y + 1 < H:  candidates.append((0, 1, self.x, self.y + 1))
        if self.x - 1 >= 0: candidates.append((-1, 0, self.x - 1, self.y))
        if self.x + 1 < W:  candidates.append((1, 0, self.x + 1, self.y))
        
        random.shuffle(candidates)

        best_move = None
        best_score = -999.0
        
        for (dx, dy, nx, ny) in candidates:
            if (nx, ny) in self.follow_history:
                continue
            if not self.is_clear(nx, ny, all_robots, collision_radius):
                continue
            alignment = (dx * vec_x) + (dy * vec_y)
            intensity = pher_attr[ny, nx]
            score = alignment + (intensity * 0.1)

            if score > best_score:
                best_score = score
                best_move = (nx, ny)
        
        if best_move is None:
            self.follow_history.clear()
            max_p = -1
            fallback = (self.x, self.y)
            for (dx, dy, nx, ny) in candidates:
                if self.is_clear(nx, ny, all_robots, collision_radius):
                    if pher_attr[ny, nx] > max_p:
                        max_p = pher_attr[ny, nx]
                        fallback = (nx, ny)
            return fallback

        self.follow_history.append(best_move)
        return best_move

    def step(self, pher_rep: np.ndarray, pher_attr: np.ndarray, targets: Set[Tuple[int, int]], 
             all_robots: List['Robot'], global_target_visits: Dict, collision_radius: int, pher_deposit: float):
        """Execute one step based on current mode."""
        H, W = pher_rep.shape
        
        self.current_terms = (0.0, 0.0, 0.0)

        if (self.x, self.y) in targets:
            if (self.x, self.y) not in self.visited_targets:
                self.visited_targets.add((self.x, self.y))
                if (self.x, self.y) not in global_target_visits:
                    global_target_visits[(self.x, self.y)] = set()
                global_target_visits[(self.x, self.y)].add(self.id)
                self.mode = "ADVERTISE"
                self.advertising_target = (self.x, self.y)
                deposit_distance_signal(pher_attr, self.x, self.y, self.x, self.y, r=5)

        if self.mode != "ADVERTISE":
            nearby_targets = []
            R = 5
            for t in targets:
                dist = abs(t[0] - self.x) + abs(t[1] - self.y)
                if dist <= R:
                    nearby_targets.append((t, dist))
            
            if nearby_targets:
                nearby_targets.sort(key=lambda x: x[1])
                target_pos = nearby_targets[0][0]
                dx = np.sign(target_pos[0] - self.x)
                dy = np.sign(target_pos[1] - self.y)
                if dx != 0 and dy != 0:
                    if random.random() < 0.5:
                        dx = 0
                    else:
                        dy = 0
                nx, ny = self.x + dx, self.y + dy
                
                if 0 <= nx < W and 0 <= ny < H and self.is_clear(nx, ny, all_robots, collision_radius):
                    self.x, self.y = nx, ny
                return

        if self.mode == "ADVERTISE":
            if self.x == self.start_x and self.y == self.start_y:
                pass
            else:
                dx = np.sign(self.start_x - self.x)
                dy = np.sign(self.start_y - self.y)
                nx, ny = self.x + dx, self.y + dy
                
                if 0 <= nx < W and 0 <= ny < H:
                    if self.is_clear(nx, ny, all_robots, collision_radius):
                        self.x, self.y = nx, ny
                        if self.advertising_target:
                            deposit_distance_signal(pher_attr, self.x, self.y, 
                                                   self.advertising_target[0], self.advertising_target[1], r=5)
        
        elif self.mode == "SEARCH":
            found_trail = False
            R = 5
            for dy in range(-R, R + 1):
                for dx in range(-R, R + 1):
                    if abs(dx) + abs(dy) <= R:
                        cx, cy = self.x + dx, self.y + dy
                        if 0 <= cx < W and 0 <= cy < H:
                            if pher_attr[cy, cx] > 1e-5:
                                found_trail = True
                                break
                if found_trail:
                    break

            if found_trail:
                self.mode = "FOLLOW"
                self.follow_history.clear()
                nx, ny = self.choose_move_follow(pher_attr, all_robots, collision_radius)
                self.x, self.y = nx, ny
            else:
                nx, ny = self.choose_move_search(pher_rep, pher_attr, all_robots, collision_radius)
                self.x, self.y = nx, ny
                deposit_uniform(pher_rep, self.x, self.y, pher_deposit, r=5)

        elif self.mode == "FOLLOW":
            nx, ny = self.choose_move_follow(pher_attr, all_robots, collision_radius)
            
            if nx == self.x and ny == self.y:
                if (nx, ny) not in targets:
                    self.mode = "SEARCH"
            
            self.x, self.y = nx, ny

        if (self.x, self.y) in targets:
            if (self.x, self.y) not in self.visited_targets:
                self.visited_targets.add((self.x, self.y))
                if (self.x, self.y) not in global_target_visits:
                    global_target_visits[(self.x, self.y)] = set()
                global_target_visits[(self.x, self.y)].add(self.id)
                self.mode = "ADVERTISE"
                self.advertising_target = (self.x, self.y)
                deposit_distance_signal(pher_attr, self.x, self.y, self.x, self.y, r=5)
