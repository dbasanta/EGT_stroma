# Reconstruction report

## Source inventory

The supplied `EGT Stromal Protection` Drive folder was searched locally and
through Google Drive. It contains the May 2026 manuscript, an older manuscript,
the figures document, a presentation, and an abstract. No Python notebook,
script, CSV, Excel workbook, or native Google Sheet containing the mouse-level
luminescence or caliper values was found under the project terms or experimental
identifiers (H3122, Alectinib, Gaussia, Cypridina, and initial resistance).

The mouse observations available in the supplied material are the individual
markers embedded in manuscript Figures 2 and 3. `scripts/digitize_mouse_data.py`
recovers these into:

- `data/digitized_composition.csv`
- `data/digitized_volume.csv`

These tables preserve the source pixel coordinate for every recovered value.
Raster-overlapping markers cannot be separated; consequently, the files contain
visible unique points rather than guaranteed original replicate counts. They
must not be described as the original raw experimental dataset.

## Equation correction supported by the results

The payoff table assigns the fibroblast payoff `alpha + k*f(t)` specifically to
interactions with sensitive cells. Population averaging therefore gives

```text
W_F = alpha*(r + s) + s*k*f(t)
```

Equation (1.1) in the prose instead prints

```text
W_F = alpha*(r + s) + k*f(t)
```

which would give fibroblasts the full wound signal even when no sensitive cells
exist. The sensitive-weighted form is supported by three independent checks:

1. It follows algebraically from the payoff table.
2. It fits the digitized composition points better than the literal printed
   equation (SSE 2.436 versus 5.338 over the recovered visible points).
3. It reproduces the published continuous and intermittent relapse times in
   Figure 6. The literal printed equation does not.

The code therefore uses the sensitive-weighted form by default and retains the
literal form under `wound_signal="manuscript"` for audit comparisons.

## Treatment-schedule reconstruction

The following rules reproduce the main treatment-comparison points:

- Initial stromal fraction: 1%.
- Initial tumor volume: 4 mm3.
- Failure: first day volume is at least 120% of baseline.
- Continuous treatment: always on.
- Intermittent treatment: one day on, one day off.
- Adaptive therapy: a hysteretic window centered on baseline,
  `[V0*(1-w), V0*(1+w)]`, with `w` searched from 1% through 20%; the window
  giving the latest relapse is retained for each initial condition.
- EMDR: `d=0.15`; no EMDR: `d=0`.

Reconstructed relapse days:

| Initial resistant | EMDR | Continuous | Adaptive | 1-on/1-off |
|---:|:---:|---:|---:|---:|
| 0% | No | 46 | 23 | 86 |
| 1% | No | 32 | 25 | 52 |
| 5% | No | 27 | 22 | 34 |
| 20% | No | 21 | 18 | 19 |
| 0% | Yes | 37 | 23 | 50 |
| 1% | Yes | 32 | 22 | 42 |
| 5% | Yes | 27 | 20 | 32 |
| 20% | Yes | 21 | 16 | 18 |

These agree with the plotted values in the manuscript figure. The complete
machine-readable table, including the selected adaptive window, is in
`results/treatment_comparison.csv`.

## Calibration interpretation

The package overlays the reported fitted parameters on the recovered visible
mouse points. It does not claim to have re-estimated the published parameter
optimum from the raster data. A new optimum obtained from visible unique points
would be statistically inappropriate because coincident mice and the raw
reporter transformations are missing. The calibration utilities are included
so the original raw tables can be inserted directly if later recovered.

The untreated-volume SSE is particularly sensitive to missing coincident points
and their baseline multiplicity. It should not be interpreted as a formal
goodness-of-fit statistic for the original experiment.

## Reproducibility

Run from the package root:

```bash
python3 scripts/digitize_mouse_data.py
python3 analysis/reproduce_manuscript_results.py
python3 -m unittest discover -s tests -v
```

The analysis produces calibration overlays, the treatment comparison,
adaptive-therapy heatmaps, CSV result tables, and
`results/reconstruction_summary.json`.
