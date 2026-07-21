from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emdr_model import (  # noqa: E402
    AdaptivePolicy,
    ContinuousPolicy,
    GameParameters,
    IntermittentPolicy,
    NoTreatmentPolicy,
    State,
    fitness,
    simulate,
    step,
)
from emdr_model.calibration import (  # noqa: E402
    CompositionObservation,
    grid_search_game,
)


class ModelTests(unittest.TestCase):
    def test_fractions_remain_on_simplex(self) -> None:
        trajectory = simulate(
            State(0.01, 0.98, 4.0), 300, IntermittentPolicy(1, 1)
        )
        np.testing.assert_allclose(
            trajectory.resistant + trajectory.sensitive + trajectory.fibroblast,
            1.0,
            atol=1e-10,
        )
        self.assertGreaterEqual(float(trajectory.fibroblast.min()), -1e-10)

    def test_no_treatment_sensitive_fitness_is_one(self) -> None:
        fit = fitness(State(0.2, 0.7, 4.0), False)
        self.assertAlmostEqual(fit.sensitive, 1.0)

    def test_resistant_fitness_is_constant(self) -> None:
        params = GameParameters(gamma=0.17)
        for on in (False, True):
            self.assertAlmostEqual(
                fitness(State(0.2, 0.7, 4.0), on, params).resistant, 0.83
            )

    def test_wound_signal_modes_differ_as_documented(self) -> None:
        state = State(0.2, 0.3, 4.0)
        literal = fitness(state, True, wound_signal="manuscript")
        weighted = fitness(state, True, wound_signal="sensitive_weighted")
        self.assertAlmostEqual(literal.fibroblast - weighted.fibroblast, 7.0 * 0.7)

    def test_zero_sensitive_exposes_manuscript_wound_signal_issue(self) -> None:
        state = State(0.5, 0.0, 4.0)
        literal = fitness(state, True, wound_signal="manuscript")
        weighted = fitness(state, True, wound_signal="sensitive_weighted")
        self.assertGreater(literal.fibroblast, weighted.fibroblast)

    def test_one_step_matches_replicator_equation(self) -> None:
        state = State(0.1, 0.8, 4.0)
        fit = fitness(state, True)
        next_state, _ = step(state, True)
        self.assertAlmostEqual(next_state.resistant, 0.1 * fit.resistant / fit.mean)
        self.assertAlmostEqual(next_state.sensitive, 0.8 * fit.sensitive / fit.mean)

    def test_policy_reset_makes_simulation_reproducible(self) -> None:
        policy = AdaptivePolicy(2.0, 4.0)
        initial = State(0.01, 0.98, 4.0)
        first = simulate(initial, 100, policy)
        second = simulate(initial, 100, policy)
        np.testing.assert_array_equal(first.treatment, second.treatment)

    def test_continuous_policy_is_always_on(self) -> None:
        trajectory = simulate(State(0.01, 0.98, 4.0), 20, ContinuousPolicy())
        self.assertTrue(bool(trajectory.treatment[:-1].all()))

    def test_no_treatment_policy_is_always_off(self) -> None:
        trajectory = simulate(State(0.01, 0.98, 4.0), 20, NoTreatmentPolicy())
        self.assertFalse(bool(trajectory.treatment.any()))

    def test_default_equation_reproduces_published_continuous_relapse_days(self) -> None:
        expected = {0.0: 37, 0.01: 32, 0.05: 27, 0.20: 21}
        for resistant, expected_day in expected.items():
            initial = State(0.99 * resistant, 0.99 * (1.0 - resistant), 4.0)
            trajectory = simulate(initial, 100, ContinuousPolicy())
            from emdr_model import relapse_day

            self.assertEqual(relapse_day(trajectory), expected_day)

    def test_intermittent_policy_preserves_block_lengths_when_starting_off(self) -> None:
        policy = IntermittentPolicy(days_on=2, days_off=3, start_on=False)
        policy.reset()
        state = State(0.1, 0.8, 4.0)
        self.assertEqual(
            [policy(day, state) for day in range(10)],
            [False, False, False, True, True] * 2,
        )

    def test_small_grid_search_recovers_generating_alpha(self) -> None:
        initial = State(0.1, 0.8, 1.0)
        fit = fitness(initial, False, GameParameters(alpha=0.6))
        next_state = State(
            initial.resistant * fit.resistant / fit.mean,
            initial.sensitive * fit.sensitive / fit.mean,
            1.0,
        )
        observations = [
            CompositionObservation("a", "g", 0, 0.1, 0.8, 0.1, False),
            CompositionObservation(
                "a",
                "g",
                1,
                next_state.resistant,
                next_state.sensitive,
                next_state.fibroblast,
                False,
            ),
        ]
        best, error = grid_search_game(
            observations,
            alpha=[0.4, 0.6, 0.8],
            d=[0.15],
            c=[0.3],
            gamma=[0.1],
            k=[7.0],
        )
        self.assertEqual(best.alpha, 0.6)
        self.assertAlmostEqual(error, 0.0)


if __name__ == "__main__":
    unittest.main()
