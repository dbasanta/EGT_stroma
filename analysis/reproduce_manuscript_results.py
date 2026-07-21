"""Reproduce calibration overlays and treatment comparisons from supplied data."""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "emdr-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "emdr-cache"))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from emdr_model import (  # noqa: E402
    AdaptivePolicy,
    ContinuousPolicy,
    GameParameters,
    IntermittentPolicy,
    NoTreatmentPolicy,
    State,
    VolumeParameters,
    relapse_day,
    simulate,
)

GROUPS = ["R_0.05pct", "R_1pct", "R_20pct", "R_0pct", "R_100pct"]
LABELS = {
    "R_0.05pct": "0.05% resistant",
    "R_1pct": "1% resistant",
    "R_20pct": "20% resistant",
    "R_0pct": "0% resistant",
    "R_100pct": "100% resistant",
}
COLORS = {"fibroblast": "#3b75af", "sensitive": "#ef8636", "resistant": "#67ca4d"}


def read_csv(name: str) -> list[dict[str, str]]:
    with (ROOT / "data" / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def baselines(rows: list[dict[str, str]]) -> dict[str, State]:
    output = {}
    for group in GROUPS:
        initial = [row for row in rows if row["group"] == group and int(row["day"]) == 0]
        means = {
            phenotype: np.mean(
                [float(row["fraction"]) for row in initial if row["phenotype"] == phenotype]
            )
            for phenotype in ("resistant", "sensitive")
        }
        output[group] = State(float(means["resistant"]), float(means["sensitive"]), 1.0)
    return output


def composition_predictions(rows: list[dict[str, str]], mode: str) -> dict[str, object]:
    initial = baselines(rows)
    output = {}
    for group in GROUPS:
        subset = [row for row in rows if row["group"] == group]
        policy = ContinuousPolicy() if subset[0]["treatment"] == "true" else NoTreatmentPolicy()
        output[group] = simulate(
            initial[group],
            max(int(row["day"]) for row in subset),
            policy,
            wound_signal=mode,  # type: ignore[arg-type]
        )
    return output


def composition_sse(rows: list[dict[str, str]], predicted: dict[str, object]) -> float:
    return float(
        sum(
            (
                float(row["fraction"])
                - getattr(predicted[row["group"]], row["phenotype"])[int(row["day"])]
            )
            ** 2
            for row in rows
        )
    )


def volume_predictions(
    composition_rows: list[dict[str, str]], volume_rows: list[dict[str, str]]
) -> dict[str, object]:
    initial_composition = baselines(composition_rows)
    output = {}
    for group in GROUPS:
        subset = [row for row in volume_rows if row["group"] == group]
        initial_volume = np.mean(
            [float(row["volume_mm3"]) for row in subset if int(row["day"]) == 0]
        )
        state = State(
            initial_composition[group].resistant,
            initial_composition[group].sensitive,
            float(initial_volume),
        )
        policy = ContinuousPolicy() if subset[0]["treatment"] == "true" else NoTreatmentPolicy()
        output[group] = simulate(state, max(int(row["day"]) for row in subset), policy)
    return output


def plot_calibrations(
    composition_rows: list[dict[str, str]], volume_rows: list[dict[str, str]], results: Path
) -> tuple[float, float, float]:
    predicted = composition_predictions(composition_rows, "sensitive_weighted")
    literal = composition_predictions(composition_rows, "manuscript")
    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    for axis, group in zip(axes.flat, GROUPS):
        trajectory = predicted[group]
        for phenotype, color in COLORS.items():
            axis.plot(trajectory.day, getattr(trajectory, phenotype), color=color, lw=2.5)
            observed = [
                row
                for row in composition_rows
                if row["group"] == group and row["phenotype"] == phenotype
            ]
            axis.scatter(
                [int(row["day"]) for row in observed],
                [float(row["fraction"]) for row in observed],
                color=color,
                edgecolor="black",
                linewidth=0.25,
                s=24,
            )
        if any(row["treatment"] == "true" for row in composition_rows if row["group"] == group):
            axis.set_facecolor("#dce6ef")
        axis.set(title=LABELS[group], xlabel="Day", ylabel="Fraction", ylim=(-0.04, 1.05))
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(
        [plt.Line2D([], [], color=value, lw=3) for value in COLORS.values()],
        [key.capitalize() for key in COLORS],
        loc="center",
    )
    figure.suptitle("Reported EGT parameters and digitized mouse composition points")
    figure.tight_layout()
    figure.savefig(results / "figure2_calibration_reconstruction.png", dpi=180)
    plt.close(figure)

    volume_predicted = volume_predictions(composition_rows, volume_rows)
    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    for axis, group in zip(axes.flat, GROUPS):
        trajectory = volume_predicted[group]
        observed = [row for row in volume_rows if row["group"] == group]
        axis.plot(trajectory.day, trajectory.volume, color="black", lw=3, label="Model")
        axis.scatter(
            [int(row["day"]) for row in observed],
            [float(row["volume_mm3"]) for row in observed],
            color="#7f170e",
            s=25,
            label="Visible mouse point",
        )
        if observed[0]["treatment"] == "true":
            axis.set_facecolor("#dce6ef")
        axis.set(title=LABELS[group], xlabel="Day", ylabel="Volume (mm³)")
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(
        [
            plt.Line2D([], [], color="black", lw=3),
            plt.Line2D([], [], color="#7f170e", marker="o", linestyle="none"),
        ],
        ["Model", "Visible mouse point"],
        loc="center",
    )
    figure.suptitle("Reported volume parameters and digitized mouse volume points")
    figure.tight_layout()
    figure.savefig(results / "figure3_volume_reconstruction.png", dpi=180)
    plt.close(figure)

    volume_error = float(
        sum(
            (
                float(row["volume_mm3"])
                - volume_predicted[row["group"]].volume[int(row["day"])]
            )
            ** 2
            for row in volume_rows
        )
    )
    return (
        composition_sse(composition_rows, predicted),
        composition_sse(composition_rows, literal),
        volume_error,
    )


def best_adaptive(initial: State, game: GameParameters, days: int = 500) -> tuple[int | None, int]:
    candidates = []
    for window in range(1, 21):
        fraction = window / 100.0
        trajectory = simulate(
            initial,
            days,
            AdaptivePolicy(
                initial.volume * (1.0 - fraction),
                initial.volume * (1.0 + fraction),
            ),
            game=game,
        )
        failure = relapse_day(trajectory)
        candidates.append((days + 1 if failure is None else failure, window))
    failure, window = max(candidates)
    return (None if failure == days + 1 else failure), window


def treatment_comparison(results: Path) -> list[dict[str, object]]:
    records = []
    for emdr, d in ((False, 0.0), (True, 0.15)):
        game = GameParameters(d=d)
        for initial_resistance in (0.0, 0.01, 0.05, 0.20):
            initial = State(
                0.99 * initial_resistance,
                0.99 * (1.0 - initial_resistance),
                4.0,
            )
            values = {
                "continuous": relapse_day(simulate(initial, 500, ContinuousPolicy(), game=game)),
                "intermittent_1_on_1_off": relapse_day(
                    simulate(initial, 500, IntermittentPolicy(1, 1), game=game)
                ),
            }
            values["adaptive"], best_window = best_adaptive(initial, game)
            for schedule, failure in values.items():
                records.append(
                    {
                        "emdr": emdr,
                        "initial_resistant_fraction": initial_resistance,
                        "schedule": schedule,
                        "relapse_day": failure,
                        "best_adaptive_window_percent": best_window if schedule == "adaptive" else "",
                    }
                )
    with (results / "treatment_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    markers = {"continuous": "o", "adaptive": "s", "intermittent_1_on_1_off": "^"}
    figure, axis = plt.subplots(figsize=(8, 6))
    for emdr, color in ((False, "#ef3123"), (True, "#1748d3")):
        for schedule, marker in markers.items():
            subset = [r for r in records if r["emdr"] is emdr and r["schedule"] == schedule]
            axis.scatter(
                [float(r["initial_resistant_fraction"]) for r in subset],
                [int(r["relapse_day"]) for r in subset],
                color=color,
                marker=marker,
                s=70,
                label=f"{'EMDR' if emdr else 'No EMDR'} — {schedule}",
            )
    axis.set_xticks([0, 0.01, 0.05, 0.20], ["0", "1", "5", "20"])
    axis.set(xlabel="Initial resistant cells (%)", ylabel="Time to relapse (days)")
    axis.legend(fontsize=8, ncol=2)
    axis.set_title("Reconstructed treatment comparison")
    figure.tight_layout()
    figure.savefig(results / "figure6_treatment_comparison.png", dpi=180)
    plt.close(figure)
    return records


def adaptive_heatmap(results: Path) -> None:
    starts = np.arange(1500.0, 4101.0, 200.0)
    windows = np.arange(1, 21)
    horizon = 2000
    records = []
    figure, axes = plt.subplots(4, 2, figsize=(12, 16), sharex=True, sharey=True)
    for row, resistance in enumerate((0.0, 0.01, 0.05, 0.20)):
        for column, (emdr, d) in enumerate(((False, 0.0), (True, 0.15))):
            matrix = np.empty((len(starts), len(windows)))
            for i, start in enumerate(starts):
                initial = State(0.99 * resistance, 0.99 * (1.0 - resistance), float(start))
                for j, window in enumerate(windows):
                    fraction = window / 100.0
                    trajectory = simulate(
                        initial,
                        horizon,
                        AdaptivePolicy(start * (1 - fraction), start * (1 + fraction)),
                        game=GameParameters(d=d),
                    )
                    failure = relapse_day(trajectory)
                    matrix[i, j] = horizon + 1 if failure is None else failure
                    records.append(
                        {
                            "initial_resistant_fraction": resistance,
                            "emdr": emdr,
                            "starting_volume_mm3": start,
                            "window_percent": int(window),
                            "relapse_day": "indefinite_within_horizon" if failure is None else failure,
                            "horizon_days": horizon,
                        }
                    )
            image = axes[row, column].imshow(
                np.log10(matrix),
                origin="lower",
                aspect="auto",
                extent=[0.5, 20.5, starts[0] - 100, starts[-1] + 100],
                vmin=np.log10(15),
                vmax=np.log10(horizon + 1),
                cmap="viridis",
            )
            axes[row, column].set_title(
                f"{resistance * 100:g}% resistant — {'EMDR' if emdr else 'No EMDR'}"
            )
            axes[row, column].set(xlabel="Centered window (%)", ylabel="Starting volume (mm³)")
    bar = figure.colorbar(image, ax=axes, shrink=0.62)
    bar.set_label("log10 time to relapse")
    figure.suptitle("Adaptive-therapy sensitivity reconstruction")
    figure.savefig(results / "supplement_adaptive_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    with (results / "adaptive_heatmap.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    composition_rows = read_csv("digitized_composition.csv")
    volume_rows = read_csv("digitized_volume.csv")
    weighted_sse, literal_sse, volume_error = plot_calibrations(
        composition_rows, volume_rows, results
    )
    comparison = treatment_comparison(results)
    adaptive_heatmap(results)
    summary = {
        "data_provenance": {
            "composition_rows": len(composition_rows),
            "volume_rows": len(volume_rows),
            "source": "Digitized visible markers from embedded manuscript Figures 2 and 3",
            "limitation": "Coincident raster markers cannot be separated; rows are visible unique points, not original replicate counts.",
        },
        "reported_game_parameters": GameParameters().__dict__,
        "reported_volume_parameters": VolumeParameters().__dict__,
        "composition_sse_sensitive_weighted": weighted_sse,
        "composition_sse_literal_printed_equation": literal_sse,
        "volume_sse_visible_points": volume_error,
        "treatment_comparison": comparison,
    }
    (results / "reconstruction_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
