import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt


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
        # print(f"Iteration {_}, step {step}, sizes {sizes}, lambdas {lambdas}")
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

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # --- parameters ---
    n_points = 1000
    k = 5
    target = n_points // k

    # --- generate random points in 2D ---
    points = rng.uniform(0, 100, size=(n_points, 2))

    # --- initialize centers randomly from points ---
    centers = points[rng.choice(n_points, size=k, replace=False)]

    print(centers.shape)
    # --- run assignment ---
    labels, lambdas = balanced_power_diagram_assign(
        points=points,
        centers=centers,
        target=target,
        iters=100,
        step0=0.1,
        decay=0.99,
        rng=rng
    )
    print(lambdas)

    # --- analyze results ---
    sizes = np.bincount(labels, minlength=k)

    print("Target size per cluster:", target)
    print("Actual cluster sizes:", sizes)
    print("Lambda values:", lambdas)

    # --- visualize ---
    # plt.figure(figsize=(6, 6))

    # for j in range(k):
    #     cluster_points = points[labels == j]
    #     plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=10, label=f"Cluster {j}")

    # plt.scatter(centers[:, 0], centers[:, 1],
    #             c='black', marker='x', s=100, label='Centers')

    # plt.title("Balanced Power Diagram Assignment")
    # plt.legend()
    # plt.grid(True)
    # plt.show()