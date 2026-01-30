import numpy as np
from typing import Tuple


def kpp_init(points: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Initialize k cluster centers using k-means++ algorithm."""
    n = points.shape[0]
    centers = np.empty((k, 2), dtype=float)
    idx = rng.integers(n)
    centers[0] = points[idx]
    d2 = np.full(n, np.inf)
    for i in range(1, k):
        d2 = np.minimum(d2, np.sum((points - centers[i-1])**2, axis=1))
        probs = d2 / d2.sum()
        idx = rng.choice(n, p=probs)
        centers[i] = points[idx]
    return centers


def balanced_power_diagram_assign(points: np.ndarray, centers: np.ndarray, target: int,
                                   iters: int, step0: float, decay: float,
                                   rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Assign points to centers using balanced power diagram (capacitated Voronoi)."""
    k = centers.shape[0]
    lambdas = np.zeros(k, dtype=float)
    labels = np.zeros(points.shape[0], dtype=int)
    step = step0
    for _ in range(iters):
        diffs = points[:, None, :] - centers[None, :, :]
        d2 = np.sum(diffs**2, axis=2)
        costs = d2 + lambdas[None, :]
        labels = np.argmin(costs, axis=1)
        sizes = np.bincount(labels, minlength=k).astype(float)
        lambdas += step * (sizes - target)
        step *= decay
    return labels, lambdas


def lloyd_balanced(points: np.ndarray, k: int, max_iters_centers: int, max_iters_assign: int,
                   step0: float, decay: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Lloyd's algorithm with balanced partitioning (equal-area Voronoi)."""
    centers = kpp_init(points, k, rng)
    target = (points.shape[0] // k)
    for _ in range(max_iters_centers):
        labels, _ = balanced_power_diagram_assign(points, centers, target, max_iters_assign, step0, decay, rng)
        new_centers = np.empty_like(centers)
        for j in range(k):
            idx = np.where(labels == j)[0]
            new_centers[j] = points[idx].mean(axis=0) if len(idx) > 0 else points[rng.integers(points.shape[0])]
        if np.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers
    labels, _ = balanced_power_diagram_assign(points, centers, target, max_iters_assign, step0, decay, rng)
    return labels, centers
