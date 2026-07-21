"""Calibration and experimental-data processing utilities.

These routines preserve the manuscript's least-squares logic. They do not
replace a full statistical observation model or identifiability analysis.
"""

from __future__ import annotations

import csv
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .core import (
    GameParameters,
    State,
    VolumeParameters,
    fitness,
    growth_rate,
    step,
)


@dataclass(frozen=True)
class LuminescenceObservation:
    tumor_id: str
    group: str
    day: int
    resistant_luminescence: float
    sensitive_luminescence: float
    volume: float


@dataclass(frozen=True)
class CompositionObservation:
    tumor_id: str
    group: str
    day: int
    resistant: float
    sensitive: float
    fibroblast: float
    treatment: bool


@dataclass(frozen=True)
class VolumeObservation:
    tumor_id: str
    group: str
    day: int
    volume: float
    resistant: float
    sensitive: float
    treatment: bool


def luminescence_conversion_factor(
    initial_resistant_fraction: float,
    resistant_luminescence: float,
    sensitive_luminescence: float,
) -> float:
    """Manuscript equation (2.1), scaling sensitive reporter to resistant reporter."""

    if not 0.0 < initial_resistant_fraction < 1.0:
        raise ValueError("Initial resistant fraction must lie strictly between 0 and 1.")
    if resistant_luminescence <= 0.0 or sensitive_luminescence <= 0.0:
        raise ValueError("Initial reporter measurements must be positive.")
    initial_sensitive_fraction = 1.0 - initial_resistant_fraction
    return (
        initial_sensitive_fraction
        / initial_resistant_fraction
        * resistant_luminescence
        / sensitive_luminescence
    )


def process_tumor_series(
    rows: Sequence[LuminescenceObservation],
    injection_resistant_fraction: float,
    injection_day: int = -17,
    treatment_start_day: int = 0,
    initial_tumor_fraction: float = 0.99,
) -> list[CompositionObservation]:
    """Apply manuscript equations (2.1)--(2.9) to one tumor's time series."""

    if not rows:
        return []
    by_day = {row.day: row for row in rows}
    if injection_day not in by_day or treatment_start_day not in by_day:
        raise ValueError("Series requires injection-day and treatment-start observations.")
    injection = by_day[injection_day]
    baseline = by_day[treatment_start_day]
    m = luminescence_conversion_factor(
        injection_resistant_fraction,
        injection.resistant_luminescence,
        injection.sensitive_luminescence,
    )
    denominator0 = baseline.resistant_luminescence + m * baseline.sensitive_luminescence
    if denominator0 <= 0.0:
        raise ValueError("Baseline transformed luminescence must be positive.")
    tumor_volume0 = initial_tumor_fraction * baseline.volume

    processed: list[CompositionObservation] = []
    for row in sorted(rows, key=lambda value: value.day):
        denominator = row.resistant_luminescence + m * row.sensitive_luminescence
        if denominator <= 0.0 or row.volume <= 0.0:
            raise ValueError("Reporter totals and volumes must be positive.")
        resistant_tumor_fraction = row.resistant_luminescence / denominator
        sensitive_tumor_fraction = m * row.sensitive_luminescence / denominator
        relative_growth = denominator / denominator0
        tumor_volume = relative_growth * tumor_volume0
        fibroblast_volume = row.volume - tumor_volume
        processed.append(
            CompositionObservation(
                tumor_id=row.tumor_id,
                group=row.group,
                day=row.day,
                resistant=resistant_tumor_fraction * tumor_volume / row.volume,
                sensitive=sensitive_tumor_fraction * tumor_volume / row.volume,
                fibroblast=fibroblast_volume / row.volume,
                treatment=row.day >= treatment_start_day,
            )
        )
    return processed


def read_composition_csv(path: str | Path) -> list[CompositionObservation]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return [
            CompositionObservation(
                tumor_id=row["tumor_id"],
                group=row["group"],
                day=int(row["day"]),
                resistant=float(row["resistant"]),
                sensitive=float(row["sensitive"]),
                fibroblast=float(row["fibroblast"]),
                treatment=row["treatment"].strip().lower() in {"1", "true", "yes"},
            )
            for row in csv.DictReader(stream)
        ]


def _composition_sse(
    observations: Sequence[CompositionObservation],
    params: GameParameters,
    wound_signal: str,
) -> float:
    """Group-mean initialization and individual-observation RSS as described."""

    total = 0.0
    groups = sorted({observation.group for observation in observations})
    for group in groups:
        subset = [observation for observation in observations if observation.group == group]
        first_day = min(observation.day for observation in subset)
        initial_rows = [observation for observation in subset if observation.day == first_day]
        state = State(
            resistant=float(np.mean([row.resistant for row in initial_rows])),
            sensitive=float(np.mean([row.sensitive for row in initial_rows])),
            volume=1.0,
        )
        predicted = {first_day: state}
        by_day: dict[int, list[CompositionObservation]] = {}
        for row in subset:
            by_day.setdefault(row.day, []).append(row)
        for day in range(first_day, max(by_day)):
            prior_rows = [row for row in subset if row.day <= day]
            representative = max(prior_rows, key=lambda row: row.day)
            fit = fitness(
                state, representative.treatment, params, wound_signal  # type: ignore[arg-type]
            )
            state = State(
                state.resistant * fit.resistant / fit.mean,
                state.sensitive * fit.sensitive / fit.mean,
                1.0,
            )
            predicted[day + 1] = state
        for row in subset:
            estimate = predicted[row.day]
            total += (row.resistant - estimate.resistant) ** 2
            total += (row.sensitive - estimate.sensitive) ** 2
            total += (row.fibroblast - estimate.fibroblast) ** 2
    return float(total)


def grid_search_game(
    observations: Sequence[CompositionObservation],
    alpha: Iterable[float],
    d: Iterable[float],
    c: Iterable[float],
    gamma: Iterable[float],
    k: Iterable[float],
    wound_signal: str = "sensitive_weighted",
) -> tuple[GameParameters, float]:
    """Exhaustive composition-parameter grid search."""

    best: GameParameters | None = None
    best_error = np.inf
    for values in itertools.product(alpha, d, c, gamma, k):
        candidate = GameParameters(*map(float, values))
        error = _composition_sse(observations, candidate, wound_signal)
        if error < best_error:
            best, best_error = candidate, error
    if best is None:
        raise ValueError("All parameter grids must be nonempty.")
    return best, float(best_error)


def volume_sse(
    observations: Sequence[VolumeObservation],
    volume: VolumeParameters,
    game: GameParameters = GameParameters(),
    wound_signal: str = "sensitive_weighted",
) -> float:
    """Group-mean trajectory RSS following equations (1.1)--(3.4).

    The first observed day in each experimental group initializes composition
    and volume using group means, as described in the manuscript. The model is
    advanced daily and compared with every individual volume observation.
    """

    total = 0.0
    groups = sorted({row.group for row in observations})
    for group in groups:
        rows = [row for row in observations if row.group == group]
        first_day = min(row.day for row in rows)
        last_day = max(row.day for row in rows)
        initial_rows = [row for row in rows if row.day == first_day]
        state = State(
            resistant=float(np.mean([row.resistant for row in initial_rows])),
            sensitive=float(np.mean([row.sensitive for row in initial_rows])),
            volume=float(np.mean([row.volume for row in initial_rows])),
        )
        predicted = {first_day: state.volume}
        for day in range(first_day, last_day):
            prior_rows = [row for row in rows if row.day <= day]
            on = max(prior_rows, key=lambda row: row.day).treatment
            state, _ = step(
                state,
                on,
                game=game,
                volume=volume,
                wound_signal=wound_signal,  # type: ignore[arg-type]
            )
            predicted[day + 1] = state.volume
        for row in rows:
            total += (row.volume - predicted[row.day]) ** 2
    return float(total)


def grid_search_volume(
    observations: Sequence[VolumeObservation],
    x: Iterable[float],
    y: Iterable[float],
    z: Iterable[float],
    treatment: bool,
    game: GameParameters = GameParameters(),
    base: VolumeParameters = VolumeParameters(),
    wound_signal: str = "sensitive_weighted",
) -> tuple[VolumeParameters, float]:
    """Search the three treatment-specific or untreated volume coefficients."""

    selected = [row for row in observations if row.treatment is treatment]
    if not selected:
        raise ValueError("No observations match the requested treatment status.")
    best: VolumeParameters | None = None
    best_error = np.inf
    for x_value, y_value, z_value in itertools.product(x, y, z):
        values = dict(base.__dict__)
        suffix = "treatment" if treatment else "no_treatment"
        values[f"x_{suffix}"] = float(x_value)
        values[f"y_{suffix}"] = float(y_value)
        values[f"z_{suffix}"] = float(z_value)
        candidate = VolumeParameters(**values)
        error = volume_sse(selected, candidate, game, wound_signal)
        if error < best_error:
            best, best_error = candidate, error
    if best is None:
        raise ValueError("All parameter grids must be nonempty.")
    return best, float(best_error)
