"""Recover plotted mouse observations from manuscript Figures 2 and 3.

The original numerical tables are not present in the supplied project folder.
This script detects the centers of visible colored markers in the embedded PNG
figures. Coincident markers cannot be separated, so the resulting CSV files
contain *visible unique points*, not guaranteed original replicate counts.

Pixel-to-data mappings were set from the plot axes. The uncertainty is roughly
half a pixel: about 0.0017 for composition, 0.033 mm^3 for treated volumes,
1.1 mm^3 for untreated-sensitive volumes, and 3.85 mm^3 for resistant volumes.
Anti-aliasing and coincident markers are larger potential error sources.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Panel:
    group: str
    treatment: bool
    day_x: tuple[tuple[int, int], ...]
    y_min_pixel: float
    y_max_pixel: float
    value_at_min_pixel: float
    value_at_max_pixel: float

    def pixel_to_value(self, y: int) -> float:
        fraction = (y - self.y_min_pixel) / (self.y_max_pixel - self.y_min_pixel)
        return self.value_at_min_pixel + fraction * (
            self.value_at_max_pixel - self.value_at_min_pixel
        )


COMPOSITION_PANELS = (
    Panel(
        "R_0.05pct",
        True,
        ((0, 80), (7, 140), (14, 203), (21, 266), (28, 327), (35, 390), (45, 479)),
        72,
        371,
        1.0,
        0.0,
    ),
    Panel(
        "R_1pct",
        True,
        ((0, 665), (7, 743), (14, 823), (21, 903), (28, 983), (35, 1065)),
        72,
        371,
        1.0,
        0.0,
    ),
    Panel(
        "R_20pct",
        True,
        ((0, 1248), (7, 1363), (14, 1481), (24, 1647)),
        72,
        371,
        1.0,
        0.0,
    ),
)


VOLUME_PANELS = (
    Panel(
        "R_0.05pct",
        True,
        ((0, 65), (7, 116), (14, 166), (21, 216), (28, 266), (35, 317), (45, 388)),
        47,
        167,
        8.0,
        0.0,
    ),
    Panel(
        "R_1pct",
        True,
        ((0, 502), (7, 566), (14, 629), (21, 694), (28, 759), (35, 823)),
        47,
        167,
        8.0,
        0.0,
    ),
    Panel(
        "R_20pct",
        True,
        ((0, 937), (7, 1031), (14, 1125), (24, 1259)),
        47,
        167,
        8.0,
        0.0,
    ),
    Panel(
        "R_0pct",
        False,
        ((0, 68), (7, 230), (14, 389)),
        420,
        509,
        200.0,
        0.0,
    ),
    Panel(
        "R_100pct",
        False,
        ((0, 941), (7, 1047), (14, 1155), (21, 1262)),
        405,
        509,
        800.0,
        0.0,
    ),
)


def color_mask(pixels: np.ndarray, color: str) -> np.ndarray:
    red, green, blue = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
    if color == "fibroblast":
        return (blue > 180) & (red < 40) & (green < 80)
    if color == "sensitive":
        return (red > 180) & (green < 120) & (blue < 100)
    if color == "resistant":
        return (red < 100) & (green > 70) & (green < 180) & (blue < 100)
    if color == "volume":
        return (red > 80) & (red < 180) & (green < 70) & (blue < 60)
    raise ValueError(color)


def marker_centers(
    mask: np.ndarray,
    x: int,
    y_bounds: tuple[int, int],
    score_threshold: float,
    half_width: int = 5,
    minimum_separation: int = 5,
) -> list[int]:
    """Locate vertical density peaks produced by circular point markers."""

    y0, y1 = y_bounds
    density = mask[y0:y1, x - half_width : x + half_width + 1].sum(axis=1)
    score = np.convolve(density.astype(float), np.ones(3), mode="same")
    candidates = [
        i
        for i in range(1, len(score) - 1)
        if score[i] >= score_threshold
        and score[i] >= score[i - 1]
        and score[i] > score[i + 1]
    ]
    selected: list[int] = []
    for candidate in sorted(candidates, key=lambda value: score[value], reverse=True):
        if all(abs(candidate - existing) >= minimum_separation for existing in selected):
            selected.append(candidate)
    return sorted(y0 + value for value in selected)


def write_composition(figure: Path, destination: Path) -> None:
    pixels = np.array(Image.open(figure).convert("RGB"))
    rows: list[dict[str, object]] = []
    for panel in COMPOSITION_PANELS:
        for phenotype in ("fibroblast", "sensitive", "resistant"):
            mask = color_mask(pixels, phenotype)
            for day, x in panel.day_x:
                centers = marker_centers(mask, x, (60, 375), 18)
                for visible_point, y in enumerate(centers, start=1):
                    value = float(np.clip(panel.pixel_to_value(y), 0.0, 1.0))
                    rows.append(
                        {
                            "group": panel.group,
                            "day": day,
                            "phenotype": phenotype,
                            "visible_point": visible_point,
                            "fraction": f"{value:.6f}",
                            "treatment": str(panel.treatment).lower(),
                            "source_pixel_y": y,
                        }
                    )

    # Baseline fibroblast markers coincide with zero-resistance markers in the
    # first two panels and are not separately detectable. Add the manuscript's
    # stated 1% baseline stroma assumption and the known injection fractions.
    for group, resistant in (("R_0.05pct", 0.0005), ("R_1pct", 0.01)):
        rows = [
            row
            for row in rows
            if not (row["group"] == group and row["day"] == 0)
        ]
        for phenotype, fraction in (
            ("fibroblast", 0.01),
            ("sensitive", 0.99 * (1.0 - resistant)),
            ("resistant", 0.99 * resistant),
        ):
            rows.append(
                {
                    "group": group,
                    "day": 0,
                    "phenotype": phenotype,
                    "visible_point": 1,
                    "fraction": f"{fraction:.6f}",
                    "treatment": "true",
                    "source_pixel_y": "assumption",
                }
            )

    # Homogeneous untreated groups are described in the text rather than fitted
    # from colored markers in Figure 2.
    for group, resistant in (("R_0pct", 0.0), ("R_100pct", 0.99)):
        for day in ((0, 7, 14) if group == "R_0pct" else (0, 7, 14, 21)):
            for phenotype, fraction in (
                ("fibroblast", 0.01),
                ("sensitive", 0.99 - resistant),
                ("resistant", resistant),
            ):
                rows.append(
                    {
                        "group": group,
                        "day": day,
                        "phenotype": phenotype,
                        "visible_point": 1,
                        "fraction": f"{fraction:.6f}",
                        "treatment": "false",
                        "source_pixel_y": "text_assumption",
                    }
                )

    rows.sort(key=lambda row: (str(row["group"]), int(row["day"]), str(row["phenotype"]), int(row["visible_point"])))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_volume(figure: Path, destination: Path) -> None:
    pixels = np.array(Image.open(figure).convert("RGB"))
    mask = color_mask(pixels, "volume")
    rows: list[dict[str, object]] = []
    for panel in VOLUME_PANELS:
        y_bounds = (45, 168) if panel.treatment else (390, 515)
        for day, x in panel.day_x:
            centers = marker_centers(mask, x, y_bounds, 15)
            for visible_point, y in enumerate(centers, start=1):
                value = max(0.0, panel.pixel_to_value(y))
                rows.append(
                    {
                        "group": panel.group,
                        "day": day,
                        "visible_point": visible_point,
                        "volume_mm3": f"{value:.6f}",
                        "treatment": str(panel.treatment).lower(),
                        "source_pixel_y": y,
                    }
                )
    rows.sort(key=lambda row: (str(row["group"]), int(row["day"]), int(row["visible_point"])))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    figure2 = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "source_figures" / "figure2.png"
    figure3 = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "source_figures" / "figure3.png"
    write_composition(figure2, ROOT / "data" / "digitized_composition.csv")
    write_volume(figure3, ROOT / "data" / "digitized_volume.csv")
    print("Wrote data/digitized_composition.csv and data/digitized_volume.csv")


if __name__ == "__main__":
    main()
