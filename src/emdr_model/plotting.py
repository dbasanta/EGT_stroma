"""Plot helpers kept separate from model calculations."""

from __future__ import annotations

from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .core import Trajectory


def plot_trajectories(
    trajectories: Mapping[str, Trajectory],
    path: str,
    failure_multiplier: float = 1.2,
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for label, trajectory in trajectories.items():
        axes[0].plot(trajectory.day, trajectory.volume, label=label, linewidth=2)
        axes[1].plot(trajectory.day, trajectory.resistant, label=f"{label}: R")
        axes[1].plot(
            trajectory.day,
            trajectory.sensitive,
            linestyle="--",
            label=f"{label}: S",
        )
    baseline = next(iter(trajectories.values())).volume[0]
    axes[0].axhline(
        failure_multiplier * baseline,
        color="black",
        linestyle=":",
        label="120% failure threshold",
    )
    axes[0].set_ylabel("Tumor volume (mm³)")
    axes[1].set_ylabel("Population fraction")
    axes[1].set_xlabel("Day")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
