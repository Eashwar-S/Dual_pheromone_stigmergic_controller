"""Utilities for launching Matplotlib-based 2D swarm simulations."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


__all__ = [
    "compute_fps",
    "make_writer",
    "run_animation",
    "FrameWriter",
]



def compute_fps(interval_ms: int) -> int:
    """Return frames-per-second for a given animation interval in milliseconds."""
    if interval_ms <= 0:
        raise ValueError("interval_ms must be positive")
    return max(1, int(round(1000 / interval_ms)))


def make_writer(interval_ms: int, *, title: str, artist: str, bitrate: int = 1800) -> FFMpegWriter:
    """Create an :class:`~matplotlib.animation.FFMpegWriter` configured for the interval."""
    fps = compute_fps(interval_ms)
    metadata = {"title": title, "artist": artist}
    return FFMpegWriter(fps=fps, metadata=metadata, bitrate=bitrate)


def run_animation(
    fig: plt.Figure,
    update: Callable[[int], object],
    *,
    interval_ms: int,
    frames: Optional[int] = None,
    blit: bool = False,
) -> FuncAnimation:
    """Launch a :class:`~matplotlib.animation.FuncAnimation` with shared defaults."""
    return FuncAnimation(fig, update, frames=frames, interval=interval_ms, blit=blit)


@dataclass
class FrameWriter:
    """Helper that saves individual frames to disk with incrementing filenames."""

    directory: str
    dpi: int = 300
    prefix: str = "frame_"
    counter: int = 0

    def __post_init__(self) -> None:
        os.makedirs(self.directory, exist_ok=True)

    def save(self, fig: plt.Figure) -> str:
        """Save ``fig`` to the managed directory and return the file path."""
        filename = f"{self.prefix}{self.counter:05d}.png"
        path = os.path.join(self.directory, filename)
        fig.savefig(path, dpi=self.dpi)
        self.counter += 1
        return path
