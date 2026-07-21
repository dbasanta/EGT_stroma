"""Demonstrate the reconstructed equations; this is not an exact paper figure."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "emdr-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "emdr-cache"))
sys.path.insert(0, str(ROOT / "src"))

from emdr_model import (  # noqa: E402
    AdaptivePolicy,
    ContinuousPolicy,
    IntermittentPolicy,
    State,
    relapse_day,
    simulate,
)
from emdr_model.plotting import plot_trajectories  # noqa: E402


def main() -> None:
    initial = State(resistant=0.001, sensitive=0.989, volume=4.0)
    days = 500
    policies = {
        "continuous": ContinuousPolicy(),
        "intermittent 1 on / 1 off": IntermittentPolicy(1, 1),
        # The manuscript does not report the exact adaptive window. These
        # illustrative thresholds turn treatment off after a 50% reduction.
        "adaptive 50--100%": AdaptivePolicy(lower=2.0, upper=4.0),
    }
    trajectories = {
        name: simulate(initial, days, policy, wound_signal="sensitive_weighted")
        for name, policy in policies.items()
    }
    for name, trajectory in trajectories.items():
        print(f"{name:27s} relapse day: {relapse_day(trajectory)}")
    output = ROOT / "emdr_model_demo.png"
    plot_trajectories(trajectories, str(output))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
