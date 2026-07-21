"""Treatment policies for the reconstructed EMDR model."""

from __future__ import annotations

from dataclasses import dataclass, field

from .core import State


@dataclass
class ContinuousPolicy:
    """Treatment is applied on every simulated day."""

    def reset(self) -> None:
        pass

    def __call__(self, day: int, state: State) -> bool:
        return True


@dataclass
class NoTreatmentPolicy:
    """Treatment is never applied."""

    def reset(self) -> None:
        pass

    def __call__(self, day: int, state: State) -> bool:
        return False


@dataclass
class IntermittentPolicy:
    """Repeat ``days_on`` treatment days followed by ``days_off`` untreated days."""

    days_on: int = 1
    days_off: int = 1
    start_on: bool = True

    def __post_init__(self) -> None:
        if self.days_on < 1 or self.days_off < 1:
            raise ValueError("days_on and days_off must both be positive.")

    def reset(self) -> None:
        pass

    def __call__(self, day: int, state: State) -> bool:
        cycle = self.days_on + self.days_off
        phase = day % cycle
        if self.start_on:
            return phase < self.days_on
        return phase >= self.days_off


@dataclass
class AdaptivePolicy:
    """Volume-triggered therapy with explicit hysteresis thresholds.

    Treatment switches on when volume is at or above ``upper`` and switches off
    when volume is at or below ``lower``. Between thresholds, its current state
    is retained.
    """

    lower: float
    upper: float
    initially_on: bool = True
    _on: bool = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.lower < 0.0 or self.upper <= self.lower:
            raise ValueError("Adaptive thresholds require 0 <= lower < upper.")
        self.reset()

    def reset(self) -> None:
        self._on = self.initially_on

    def __call__(self, day: int, state: State) -> bool:
        if self._on and state.volume <= self.lower:
            self._on = False
        elif not self._on and state.volume >= self.upper:
            self._on = True
        return self._on
