# ========= Lloyd convergence trace + animation =========
import matplotlib.animation as animation
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from centralized_approach import *

def lloyd_balanced_with_trace(points: np.ndarray,
                              k: int,
                              max_iters_centers: int = 20,
                              max_iters_assign: int = 40,
                              step0: float = 0.5,
                              decay: float = 0.9,
                              rng=None):
    """
    Run the balanced Lloyd algorithm but record (centers, labels) at each iteration
    for visualization. Frame 0 corresponds to the initial k-means++ centers.

    Returns
    -------
    history : List[Tuple[np.ndarray, np.ndarray]]
        A list of (centers, labels) for each recorded iteration.
        centers: (k, 2) array
        labels:  (N,) array of ints in [0, k-1]
    """
    if rng is None:
        # use module-level rng if present; else make a local one
        try:
            _rng = globals().get("rng", None)
        except Exception:
            _rng = None
        rng = _rng or np.random.default_rng(0)

    N = points.shape[0]
    target = N // k

    centers = kpp_init(points, k, rng=rng).astype(float)
    history = []

    # Initial assignment with current centers (this is the "before Lloyd" frame)
    labels, _ = balanced_power_diagram_assign(
        points, centers, target=target, iters=max_iters_assign,
        step0=step0, decay=decay, rng=rng
    )
    history.append((centers.copy(), labels.copy()))

    for _ in range(max_iters_centers):
        # Recompute centers from current labels
        new_centers = centers.copy()
        for i in range(k):
            idx = (labels == i)
            if np.any(idx):
                new_centers[i] = points[idx].mean(axis=0)
            else:
                # reseed empty center to a random point
                new_centers[i] = points[rng.integers(0, N)]

        # If centers stopped moving, still append and stop
        history.append((new_centers.copy(), labels.copy()))
        if np.allclose(new_centers, centers, atol=1e-6, rtol=1e-6):
            centers = new_centers
            break

        centers = new_centers
        # Re-assign with updated centers
        labels, _ = balanced_power_diagram_assign(
            points, centers, target=target, iters=max_iters_assign,
            step0=step0, decay=decay, rng=rng
        )

    # Ensure last frame includes final labels for visual consistency
    if history[-1][1] is not labels:
        history[-1] = (history[-1][0], labels.copy())

    return history


def animate_lloyd_convergence(points: np.ndarray,
                              history,
                              figsize=(6, 6),
                              interval_ms: int = 800,
                              save_path: str = None,
                              dpi: int = 120,
                              title_prefix: str = "Balanced Lloyd"):
    """
    Create a Matplotlib animation that shows cluster labels and moving centers.

    Parameters
    ----------
    points : (N, 2) array
        Input points.
    history : list of (centers, labels)
        Output of lloyd_balanced_with_trace.
    figsize : tuple
        Figure size.
    interval_ms : int
        Delay between frames.
    save_path : str or None
        If provided, save animation ('.mp4' or '.gif') instead of just showing.
    dpi : int
        Resolution for saving.
    title_prefix : str
        Text prefix for the figure title.

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        The constructed animation object.
    """
    centers_seq = np.array([h[0] for h in history])   # (T, k, 2)
    labels_seq  = [h[1] for h in history]             # list of (N,) arrays
    T, k, _ = centers_seq.shape
    N = points.shape[0]

    # Color palette for k clusters
    palette = [cm.get_cmap("tab20")(i % 20) for i in range(k)]

    def labels_to_colors(lbls):
        return [palette[int(li)] for li in lbls]

    # Figure & axes
    fig, ax = plt.subplots(figsize=figsize)
    xmin, ymin = points.min(axis=0) - 0.5
    xmax, ymax = points.max(axis=0) + 0.5
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Scatter for points (colors updated per frame)
    scat_pts = ax.scatter(points[:, 0], points[:, 1], s=10,
                          c=labels_to_colors(labels_seq[0]),
                          linewidths=0.0)

    # Centers (X markers)
    centers0 = centers_seq[0]
    scat_centers = ax.scatter(centers0[:, 0], centers0[:, 1],
                              s=150, marker='X', edgecolors='k',
                              linewidths=1.2, zorder=5,
                              c=[palette[j] for j in range(k)])

    # Lines to show center trajectories
    lines = [ax.plot([], [], lw=1.5, alpha=0.9, color=palette[j])[0] for j in range(k)]

    # Annotation for iteration + SSE
    title = ax.set_title(f"{title_prefix}: iter 0 / {T-1} | SSE: …")

    def frame_sse(centers, labels):
        dif = points - centers[labels]
        return float(np.sum(np.einsum('ij,ij->i', dif, dif)))

    def init():
        # initialize lines empty
        for ln in lines:
            ln.set_data([], [])
        return (scat_pts, scat_centers, *lines, title)

    def update(t):
        centers = centers_seq[t]
        labels  = labels_seq[t]

        # Update point colors
        colors = labels_to_colors(labels)
        scat_pts.set_facecolors(colors)

        # Update center locations
        scat_centers.set_offsets(centers)

        # Update center trajectories up to frame t
        for j in range(k):
            traj = centers_seq[:t+1, j, :]
            lines[j].set_data(traj[:, 0], traj[:, 1])

        sse = frame_sse(centers, labels)
        title.set_text(f"{title_prefix}: iter {t} / {T-1} | SSE: {sse:.1f}")
        return (scat_pts, scat_centers, *lines, title)

    anim = animation.FuncAnimation(fig, update, frames=T, init_func=init,
                                   interval=interval_ms, blit=False, repeat=False)

    if save_path:
        # choose writer based on extension
        ext = save_path.lower().rsplit('.', 1)[-1]
        if ext == 'gif':
            anim.save(save_path, dpi=dpi, writer='pillow')
        else:
            # mp4 fallback (requires ffmpeg installed)
            anim.save(save_path, dpi=dpi, writer=animation.FFMpegWriter(fps=max(1, int(1000/interval_ms))))
    return anim


if __name__ == "__main__":
    # Example: build a grid of points like your main script does
    W = H = 75
    yy, xx = np.mgrid[0:H, 0:W]
    points = np.column_stack((xx.ravel() + 0.5, yy.ravel() + 0.5))
    k = 18

    # Deterministic RNG
    rng = np.random.default_rng(42)

    # Record the Lloyd iterations
    history = lloyd_balanced_with_trace(points, k,
                                        max_iters_centers=20,
                                        max_iters_assign=40,
                                        step0=0.1, decay=0.1,
                                        rng=rng)

    # Animate in a pop-up window; or pass save_path="lloyd.mp4"/"lloyd.gif" to save
    anim = animate_lloyd_convergence(points, history,
                                     figsize=(6, 6),
                                     interval_ms=600,
                                     save_path=None)  # e.g., "lloyd.mp4"

    plt.show()
