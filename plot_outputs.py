"""Save Matplotlib figures and their plotted data in reproducible formats."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _series_label(label: str, kind: str, index: int) -> str:
    """Return a readable label for a plotted object with no public label."""
    if label and not label.startswith("_"):
        return label
    return f"{kind}_{index}"


def write_figure_data_csv(figure: plt.Figure, output_path: Path) -> None:
    """Write all line curves and 2-D image values from a figure to CSV.

    Line records use ``x`` and ``y``. Image records use spatial ``x`` and ``y``
    pixel-center coordinates plus ``value``. ``panel_index`` identifies the
    Matplotlib axes; colorbar axes contain no exported data.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "panel_index",
                "panel_title",
                "data_kind",
                "series",
                "series_index",
                "point_index",
                "x",
                "y",
                "value",
            ]
        )

        for panel_index, axes in enumerate(figure.axes):
            panel_title = axes.get_title()
            for line_index, line in enumerate(axes.get_lines()):
                label = _series_label(line.get_label(), "curve", line_index)
                x_data = np.asarray(line.get_xdata()).reshape(-1)
                y_data = np.asarray(line.get_ydata()).reshape(-1)
                for point_index, (x_value, y_value) in enumerate(zip(x_data, y_data)):
                    writer.writerow(
                        [
                            panel_index,
                            panel_title,
                            "curve",
                            label,
                            line_index,
                            point_index,
                            x_value,
                            y_value,
                            "",
                        ]
                    )

            for image_index, image in enumerate(axes.get_images()):
                values = np.ma.asarray(image.get_array())
                if values.ndim != 2:
                    continue
                label = _series_label(image.get_label(), "image", image_index)
                left, right, bottom, top = image.get_extent()
                row_count, column_count = values.shape
                x_coordinates = np.linspace(left, right, column_count, endpoint=False)
                x_coordinates += (right - left) / (2 * column_count)
                y_coordinates = np.linspace(bottom, top, row_count, endpoint=False)
                y_coordinates += (top - bottom) / (2 * row_count)
                if image.origin == "upper":
                    y_coordinates = y_coordinates[::-1]

                point_index = 0
                for row_index, y_value in enumerate(y_coordinates):
                    for column_index, x_value in enumerate(x_coordinates):
                        value = values[row_index, column_index]
                        if np.ma.is_masked(value):
                            value = float("nan")
                        writer.writerow(
                            [
                                panel_index,
                                panel_title,
                                "image",
                                label,
                                image_index,
                                point_index,
                                x_value,
                                y_value,
                                value,
                            ]
                        )
                        point_index += 1


def save_figure_outputs(figure: plt.Figure, output_stem: Path, dpi: int = 300) -> None:
    """Save one figure as matching PNG, PDF, and plotted-data CSV files."""
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.parent / f"{output_stem.name}.png"
    pdf_path = output_stem.parent / f"{output_stem.name}.pdf"
    csv_path = output_stem.parent / f"{output_stem.name}.csv"
    figure.savefig(png_path, dpi=dpi)
    figure.savefig(pdf_path)
    write_figure_data_csv(figure, csv_path)
