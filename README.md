# Reconstructed EMDR evolutionary game model

This repository reconstructs the mathematical model described in
`Manuscript_May_2026.docx`. It contains:

- the three-strategy evolutionary game (fibroblast, sensitive, resistant),
- the manuscript's discrete tumor-volume model,
- continuous, intermittent, and hysteretic adaptive treatment policies,
- relapse calculations,
- grid-search calibration utilities for composition and volume data,
- a runnable example and dependency-free unit tests.

## What is exact and what is inferred

The payoff equations, fitted parameter values, volume equations, and carrying
capacity are transcribed from the manuscript. The following details were not
specified and are therefore configurable assumptions:

1. **Wound signal.** Equation (1.1) and the payoff table both give
   `W_F = alpha*(r+s) + s*k*treatment`. This sensitive-weighted form is the
   default and reproduces the published relapse times. A deliberately
   unweighted counterfactual, `W_F = alpha*(r+s) + k*treatment`, remains
   available as `wound_signal="unweighted"` for sensitivity checks.
2. **Adaptive thresholds.** The precise original treatment window is absent.
   The implementation accepts explicit lower and upper volume thresholds and
   uses standard hysteresis: treatment turns on at the upper threshold and off
   at the lower threshold.
3. **Update order.** Fitness, composition, and volume change are calculated
   from the state at the start of each day. This is the most direct reading of
   the indexed equations.
4. **Relapse.** A comparable endpoint (first volume at or above 120% of initial
   volume) is supplied for all policies. A retrospective implementation of the
   manuscript's asymmetric adaptive-therapy endpoint is also supplied.

The supplied project contains no original numerical mouse table. This package
therefore includes a reproducible digitization of the visible mouse markers in
the manuscript figures. Coincident raster markers cannot be separated, so the
CSV files are an approximate recovery rather than the original raw tables.

## Quick start

From this directory:

```bash
python3 examples/reproduce_model_demo.py
python3 scripts/digitize_mouse_data.py
python3 analysis/reproduce_manuscript_results.py
python3 -m unittest discover -s tests -v
```

The example writes `emdr_model_demo.png` and prints relapse times.

## Package layout

```text
src/emdr_model/core.py          Model equations and simulation
src/emdr_model/schedules.py     Treatment policies
src/emdr_model/calibration.py   Data processing and grid searches
src/emdr_model/plotting.py      Plot helpers
examples/                       Runnable demonstrations
analysis/                       Manuscript calibration and treatment analyses
scripts/                        Reproducible plot-digitization pipeline
source_figures/                 Original embedded mouse-data figures
data/                           Digitized visible mouse observations
results/                        Reconstructed figures, tables, and audit summary
tests/                          Numerical and invariance checks
data_templates/                 CSV schemas for recovered raw data
```

## Using the model

```python
from emdr_model import (
    AdaptivePolicy,
    GameParameters,
    State,
    VolumeParameters,
    simulate,
)

initial = State(resistant=0.001, sensitive=0.989, volume=4.0)
policy = AdaptivePolicy(lower=2.0, upper=4.0, initially_on=True)

trajectory = simulate(
    initial,
    days=500,
    policy=policy,
    game=GameParameters(),
    volume=VolumeParameters(),
    wound_signal="sensitive_weighted",
)
```

Fractions include stroma: `fibroblast = 1 - resistant - sensitive`.

## Input data formats

Templates are in `data_templates/`. Composition fitting expects one row per
tumor and time point. Volume fitting expects the observed volume plus the model
composition at that time; these compositions can be generated using the fitted
game parameters or supplied from processed experimental data.

## Important modeling caveat

The manuscript calls `(1 + g_t) * v_t` exponential decay. It is a discrete
geometric update. The code preserves the manuscript equation rather than
silently replacing it with `exp(g_t) * v_t`.
