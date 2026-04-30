import numpy as np


def deposit_uniform(pher: np.ndarray, x: int, y: int, amount: float, r: int = 5):
    """Deposit uniform pheromone in Manhattan radius r around (x, y)."""
    H, W = pher.shape
    x0, y0 = int(x), int(y)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if abs(dx) + abs(dy) <= r:
                cx, cy = x0 + dx, y0 + dy
                if 0 <= cx < W and 0 <= cy < H:
                    pher[cy, cx] += amount


def deposit_distance_signal(pher: np.ndarray, x: int, y: int, target_x: int, target_y: int, r: int = 5):
    """Deposit exponential distance-based pheromone signal pointing to target."""
    H, W = pher.shape
    x0, y0 = int(x), int(y)
    dist_to_target = np.sqrt((x0 - target_x)**2 + (y0 - target_y)**2)
    signal_strength = 10.0 / (0.1 * dist_to_target + 0.00001)
    
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if abs(dx) + abs(dy) <= r:
                cx, cy = x0 + dx, y0 + dy
                if 0 <= cx < W and 0 <= cy < H:
                    pher[cy, cx] = np.maximum(pher[cy, cx], signal_strength)


def apply_decay(pher: np.ndarray, tau_decay: float, pher_min: float = 1e-6):
    """Apply exponential decay to pheromone field."""
    decay_factor = np.exp(-1.0 / tau_decay)
    pher *= decay_factor
    pher[pher < pher_min] = 0.0
