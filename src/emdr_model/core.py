"""Core discrete-time EMDR evolutionary game and tumor-volume model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

WoundSignal = Literal["manuscript", "sensitive_weighted"]


@dataclass(frozen=True)
class GameParameters:
    """Parameters reported by the manuscript's composition grid search."""

    alpha: float = 0.95
    d: float = 0.15
    c: float = 0.30
    gamma: float = 0.10
    k: float = 7.0


@dataclass(frozen=True)
class VolumeParameters:
    """Scaling factors and carrying capacity reported in the manuscript."""

    x_treatment: float = 0.70
    y_treatment: float = 0.06
    z_treatment: float = 0.05
    x_no_treatment: float = 0.12
    y_no_treatment: float = 0.14
    z_no_treatment: float = 0.08
    carrying_capacity: float = 5000.0


@dataclass(frozen=True)
class State:
    resistant: float
    sensitive: float
    volume: float

    @property
    def fibroblast(self) -> float:
        return 1.0 - self.resistant - self.sensitive

    def validate(self, tolerance: float = 1e-10) -> None:
        values = (self.resistant, self.sensitive, self.fibroblast)
        if not all(np.isfinite(values)) or not np.isfinite(self.volume):
            raise ValueError("State values must be finite.")
        if min(values) < -tolerance:
            raise ValueError("Cell fractions must be nonnegative and sum to one.")
        if self.volume < 0.0:
            raise ValueError("Tumor volume must be nonnegative.")


@dataclass(frozen=True)
class Fitness:
    fibroblast: float
    resistant: float
    sensitive: float
    mean: float


@dataclass(frozen=True)
class Trajectory:
    """State at integer days and treatment applied over each preceding interval."""

    day: np.ndarray
    resistant: np.ndarray
    sensitive: np.ndarray
    fibroblast: np.ndarray
    volume: np.ndarray
    treatment: np.ndarray
    growth_rate: np.ndarray


class TreatmentPolicy(Protocol):
    def reset(self) -> None: ...

    def __call__(self, day: int, state: State) -> bool: ...


def _validate_parameters(game: GameParameters, volume: VolumeParameters) -> None:
    bounded = {
        "alpha": game.alpha,
        "d": game.d,
        "c": game.c,
        "gamma": game.gamma,
    }
    if any(not 0.0 <= value <= 1.0 for value in bounded.values()):
        raise ValueError("alpha, d, c, and gamma must each lie in [0, 1].")
    if game.k < 0.0:
        raise ValueError("Wound-signal benefit k must be nonnegative.")
    if volume.carrying_capacity <= 0.0:
        raise ValueError("Carrying capacity must be positive.")


def fitness(
    state: State,
    treatment: bool,
    game: GameParameters = GameParameters(),
    wound_signal: WoundSignal = "sensitive_weighted",
) -> Fitness:
    """Compute equations (1.1)--(1.4) at the current state."""

    state.validate()
    r, s, f = state.resistant, state.sensitive, state.fibroblast
    on = float(treatment)

    if wound_signal == "manuscript":
        wound = game.k * on
    elif wound_signal == "sensitive_weighted":
        wound = s * game.k * on
    else:
        raise ValueError(f"Unknown wound_signal mode: {wound_signal!r}")

    w_f = game.alpha * (r + s) + wound
    w_r = 1.0 - game.gamma
    w_s = f * (1.0 - (1.0 - game.d) * game.c * on) + (r + s) * (
        1.0 - game.c * on
    )
    mean = f * w_f + r * w_r + s * w_s
    return Fitness(w_f, w_r, w_s, mean)


def growth_rate(
    state: State,
    fit: Fitness,
    treatment: bool,
    volume: VolumeParameters = VolumeParameters(),
) -> float:
    """Compute manuscript equation (3.1) or (3.2)."""

    r, s, f = state.resistant, state.sensitive, state.fibroblast
    if treatment:
        return (
            -volume.x_treatment * s * (1.0 - fit.sensitive)
            + volume.y_treatment * r * fit.resistant
            + volume.z_treatment * f * fit.fibroblast
        )
    return (
        volume.x_no_treatment * s * fit.sensitive
        + volume.y_no_treatment * r * fit.resistant
        + volume.z_no_treatment * f * fit.fibroblast
    )


def step(
    state: State,
    treatment: bool,
    game: GameParameters = GameParameters(),
    volume: VolumeParameters = VolumeParameters(),
    wound_signal: WoundSignal = "sensitive_weighted",
) -> tuple[State, float]:
    """Advance composition and tumor volume by one manuscript time step."""

    _validate_parameters(game, volume)
    fit = fitness(state, treatment, game, wound_signal)
    if fit.mean <= 0.0 or not np.isfinite(fit.mean):
        raise ValueError("Mean fitness must be finite and positive for replication.")

    r_next = state.resistant * fit.resistant / fit.mean
    s_next = state.sensitive * fit.sensitive / fit.mean
    g = growth_rate(state, fit, treatment, volume)

    if g < 0.0:
        v_next = (1.0 + g) * state.volume
    else:
        v_next = state.volume + g * state.volume * (
            1.0 - state.volume / volume.carrying_capacity
        )

    if v_next < -1e-10:
        raise ValueError(
            f"Volume update became negative (g={g:.6g}); the discrete model is unstable."
        )
    next_state = State(float(r_next), float(s_next), max(0.0, float(v_next)))
    next_state.validate(tolerance=1e-8)
    return next_state, float(g)


def simulate(
    initial: State,
    days: int,
    policy: TreatmentPolicy,
    game: GameParameters = GameParameters(),
    volume: VolumeParameters = VolumeParameters(),
    wound_signal: WoundSignal = "sensitive_weighted",
) -> Trajectory:
    """Simulate from day zero through ``days`` inclusive."""

    if days < 0:
        raise ValueError("days must be nonnegative.")
    initial.validate()
    _validate_parameters(game, volume)
    policy.reset()

    day = np.arange(days + 1, dtype=int)
    r = np.empty(days + 1)
    s = np.empty(days + 1)
    f = np.empty(days + 1)
    v = np.empty(days + 1)
    treatment = np.zeros(days + 1, dtype=bool)
    rates = np.full(days + 1, np.nan)

    state = initial
    r[0], s[0], f[0], v[0] = (
        state.resistant,
        state.sensitive,
        state.fibroblast,
        state.volume,
    )
    for t in range(days):
        on = bool(policy(t, state))
        treatment[t] = on
        state, rates[t] = step(state, on, game, volume, wound_signal)
        r[t + 1], s[t + 1], f[t + 1], v[t + 1] = (
            state.resistant,
            state.sensitive,
            state.fibroblast,
            state.volume,
        )

    return Trajectory(day, r, s, f, v, treatment, rates)


def relapse_day(trajectory: Trajectory, multiplier: float = 1.2) -> int | None:
    """First post-baseline day with volume >= multiplier times baseline."""

    threshold = multiplier * trajectory.volume[0]
    hits = np.flatnonzero(trajectory.volume[1:] >= threshold)
    return None if hits.size == 0 else int(hits[0] + 1)


def manuscript_adaptive_failure_day(
    trajectory: Trajectory, multiplier: float = 1.2
) -> int | None:
    """Retrospective reading of the manuscript's adaptive failure definition.

    Returns the first threshold-crossing day at or after the final off-treatment
    interval, meaning treatment is continuously on thereafter within the
    simulated horizon. This endpoint depends on the selected horizon.
    """

    off = np.flatnonzero(~trajectory.treatment[:-1])
    continuously_on_from = 0 if off.size == 0 else int(off[-1] + 1)
    threshold = multiplier * trajectory.volume[0]
    candidates = np.flatnonzero(
        (trajectory.day >= continuously_on_from) & (trajectory.volume >= threshold)
    )
    return None if candidates.size == 0 else int(candidates[0])
