"""Batch analysis of sublimation image measurements.

Each measurement image is reduced to one refractive-index and porosity value
per ROI.  The input files must be named with Unix timestamps, for example
``1787033231.1361303_data.thz``.  Results are written as six PNG/PDF plots and a
CSV file so the numerical values can be reused.
"""

from __future__ import annotations

import ast
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from pydotthz import DotthzFile
from shapely.geometry import Point, Polygon

from refractive_index_image import (
    DEFAULT_EMT_MODEL,
    get_refraction_index,
    porosity_from_emt_refractive_index,
)


FREQUENCY_BAND_THz = (0.9, 1.1)


@dataclass(frozen=True)
class MeasurementResult:
    """ROI averages and metadata derived from one image measurement."""

    path: Path
    timestamp: float
    temperature_k: float
    pressure_mbar: float
    roi_values: dict[str, tuple[float, float]]  # label -> (n, porosity)


def _first_measurement(file: DotthzFile) -> str:
    keys = list(file.keys())
    if not keys:
        raise ValueError("The THz file does not contain a measurement.")
    return keys[0]


def _number(metadata: Any, key: str, path: Path, fallback_keys: tuple[str, ...] = ()) -> float:
    """Read a numeric metadata field, allowing known acquisition-name variants."""
    for candidate in (key, *fallback_keys):
        try:
            value = metadata[candidate]
        except KeyError:
            continue
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path.name} has a non-numeric {candidate!r} value: {value!r}") from exc
    alternatives = ", ".join(repr(candidate) for candidate in (key, *fallback_keys))
    raise KeyError(f"{path.name} has none of the expected metadata fields: {alternatives}")


def timestamp_from_filename(path: Path) -> float:
    """Read the Unix timestamp from a timestamp-named .thz file."""
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(?:_data)?", path.stem)
    if not match:
        raise ValueError(
            "Expected a Unix timestamp filename such as "
            f"'1787033231.2_data.thz', got {path.name!r}."
        )
    return float(match.group(1))


def roi_label_sort_key(label: str) -> tuple[int, str]:
    """Sort labels such as 'ROI 1' through 'ROI 4' in numerical order."""
    match = re.search(r"\d+", label)
    return (int(match.group()) if match else 10**9, label)


def extract_rois(path: Path, measurement_key: str | None = None) -> dict[str, np.ndarray]:
    """Return an ``(N, 2)`` array of image indices for every metadata ROI."""
    with DotthzFile(path, "r") as image_file:
        measurement_key = measurement_key or _first_measurement(image_file)
        metadata = image_file[measurement_key].metadata
        height = int(_number(metadata, "height", path))
        width = int(_number(metadata, "width", path))

        labels = [label.strip() for label in str(metadata["ROI Labels"]).split(",")]

        if len(labels) != 4:
            raise ValueError(f"Expected exactly four ROIs in {path.name}, found {len(labels)}.")

        y_grid, x_grid = np.indices((height, width))
        pixel_coordinates = np.column_stack((x_grid.ravel(), y_grid.ravel()))
        rois: dict[str, np.ndarray] = {}
        for index, label in enumerate(labels):
            raw_points = metadata[f"ROI {index}"]
            try:
                points = ast.literal_eval(raw_points) if isinstance(raw_points, str) else raw_points
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f"Unable to parse ROI {index} in {path.name}") from exc
            # ROI vertices use the acquisition coordinate system; arrays use (x, y).
            polygon = Polygon([(width - 1 - y, x) for x, y in points])
            inside = np.fromiter(
                (polygon.covers(Point(x, y)) for x, y in pixel_coordinates),
                dtype=bool,
                count=len(pixel_coordinates),
            ).reshape(height, width)
            indices = np.argwhere(inside)
            if not len(indices):
                raise ValueError(f"ROI {label!r} in {path.name} does not contain any pixels.")
            # Dataset indexing is [x, y, time], while argwhere returned [y, x].
            rois[label] = indices[:, [1, 0]]
    return rois


def resolve_thz_file(path: Path) -> Path:
    """Accept either a .thz file or a directory containing exactly one .thz file."""
    if path.is_file():
        return path
    files = sorted(path.glob("*.thz")) if path.is_dir() else []
    if len(files) == 1:
        return files[0]
    if len(files) > 1:
        raise ValueError(f"{path} contains {len(files)} .thz files; select the intended reference file explicitly.")
    raise FileNotFoundError(f"No .thz file found at {path}")


def load_reference_trace(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a single-pixel reference trace from a .thz file or its directory."""
    path = resolve_thz_file(path)
    with DotthzFile(path, "r") as file:
        for key in file.keys():
            datasets = file[key].datasets
            if "Sample" in datasets:
                sample = np.asarray(datasets["Sample"])
                return sample[:, 0], sample[:, 1]
            if "Reference" in datasets:
                reference = np.asarray(datasets["Reference"])
                return reference[:, 0], reference[:, 1]
    raise ValueError(f"No 'Sample' or 'Reference' trace found in {path}")


def solid_ice_index(solid_ice: Path, solid_ice_reference: Path, thickness_mm: float) -> float:
    time, trace = load_reference_trace(solid_ice)
    reference_time, reference_trace = load_reference_trace(solid_ice_reference)
    frequencies, _, complex_index = get_refraction_index(
        time,
        trace[np.newaxis, np.newaxis, :],
        reference_time,
        reference_trace,
        window_half_width=25,
        win_func="hanning",
        min_frequency=0.5,
        max_frequency=3.0,
        d_mm=thickness_mm,
    )
    band = (frequencies > FREQUENCY_BAND_THz[0]) & (frequencies < FREQUENCY_BAND_THz[1])
    value = float(np.nanmean(np.real(complex_index[0, 0, band])))
    if not np.isfinite(value):
        raise RuntimeError("Could not calculate the solid-ice refractive index near 1 THz.")
    return value


def analyse_measurement(path: Path, reference_time: np.ndarray, reference_trace: np.ndarray,
                        solid_index: float, thickness_mm: float) -> MeasurementResult:
    with DotthzFile(path, "r") as file:
        key = _first_measurement(file)
        datasets = file[key].datasets
        metadata = file[key].metadata
        time = np.asarray(datasets["time"])
        traces = np.asarray(datasets["dataset"])
        # Older acquisitions call the same sensor T_SH rather than T_SHC.
        temperature = _number(metadata, "T_SHC [K]", path, fallback_keys=("T_SH [K]", "T_S [K]"))
        pressure = _number(metadata, "P [mbar]", path)

    rois = extract_rois(path, key)
    values: dict[str, tuple[float, float]] = {}
    for label, coordinates in rois.items():
        roi_trace = np.nanmean(traces[coordinates[:, 0], coordinates[:, 1], :], axis=0)
        frequencies, refractive_index, _ = get_refraction_index(
            time,
            roi_trace[np.newaxis, np.newaxis, :],
            reference_time,
            reference_trace,
            window_half_width=25,
            win_func="hanning",
            min_frequency=0.5,
            max_frequency=3.0,
            d_mm=thickness_mm,
        )
        band = (frequencies > FREQUENCY_BAND_THz[0]) & (frequencies < FREQUENCY_BAND_THz[1])
        n_eff = float(np.nanmean(refractive_index[0, 0, band]))
        # Keep the same rejection threshold used by refractive_index_image.py.
        if n_eff > 2.1:
            n_eff = float("nan")
        porosity = float(porosity_from_emt_refractive_index(n_eff, solid_index, model=DEFAULT_EMT_MODEL))
        values[label] = n_eff, porosity

    return MeasurementResult(path, timestamp_from_filename(path), temperature, pressure, values)


def save_plot(results: list[MeasurementResult], labels: list[str], x_key: str, value_index: int,
              output_directory: Path) -> None:
    x_labels = {
        "time": "Time after first measurement [h]",
        "temperature": "Temperature [K]",
        "pressure": "Pressure [mbar]",
    }
    y_labels = ("Refractive index near 1 THz", "Porosity")
    suffix = "refractive_index" if value_index == 0 else "porosity"

    first_timestamp = results[0].timestamp
    x_values = {
        "time": np.array([(result.timestamp - first_timestamp) / 3600 for result in results]),
        "temperature": np.array([result.temperature_k for result in results]),
        "pressure": np.array([result.pressure_mbar for result in results]),
    }[x_key]

    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for label in labels:
        y_values = [result.roi_values[label][value_index] for result in results]
        axis.scatter(x_values, y_values, label=label)
    axis.set_xlabel(x_labels[x_key])
    axis.set_ylabel(y_labels[value_index])
    axis.grid(True, alpha=0.3)
    axis.legend(title="ROI")
    base = output_directory / f"{suffix}_vs_{x_key}"
    fig.savefig(base.with_suffix(".png"), dpi=300)
    fig.savefig(base.with_suffix(".pdf"))
    plt.show()


def write_csv(results: list[MeasurementResult], labels: list[str], output_directory: Path) -> None:
    with (output_directory / "sublimation_roi_results.csv").open("w", newline="") as output:
        columns = ["file", "timestamp", "time_after_first_measurement_h", "temperature_k", "pressure_mbar"]
        columns += [item for label in labels for item in (f"{label}_refractive_index", f"{label}_porosity")]
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        first_timestamp = results[0].timestamp
        for result in results:
            row: dict[str, float | str] = {
                "file": result.path.name,
                "timestamp": result.timestamp,
                "time_after_first_measurement_h": (result.timestamp - first_timestamp) / 3600,
                "temperature_k": result.temperature_k,
                "pressure_mbar": result.pressure_mbar,
            }
            for label in labels:
                row[f"{label}_refractive_index"], row[f"{label}_porosity"] = result.roi_values[label]
            writer.writerow(row)


def main() -> None:
    # Set all analysis inputs here before running the script.
    measurements_directory = Path("/Users/linus/Documents/porosity_august3_lr_temperature_profile_2026_focused_frost_10.0mm_7.75g/data_image")
    reference_path = Path("/Users/linus/Documents/porosity_august3_2026_focused_silicon_reference/data/trans/single_pixel/1787033231.1361303_sp_data.thz")
    solid_ice_path = Path("/Users/linus/Documents/collimated_solid_ice_3a/data/single_pixel")
    solid_ice_reference_path = Path("/Users/linus/Documents/collimated_silicon_metal_sheet_fix_focus/data/single_pixel")
    thickness_mm = 10.0
    # Set this to a Path(...) to override the default output location.
    output_directory: Path | None = None

    files = sorted(measurements_directory.glob("*.thz"), key=timestamp_from_filename)
    if not files:
        raise FileNotFoundError(f"No .thz files found in {measurements_directory}")
    output_directory = output_directory or measurements_directory / "sublimation_analysis"
    output_directory.mkdir(parents=True, exist_ok=True)

    reference_time, reference_trace = load_reference_trace(reference_path)
    solid_index = solid_ice_index(solid_ice_path, solid_ice_reference_path, thickness_mm)
    results = [
        analyse_measurement(path, reference_time, reference_trace, solid_index, thickness_mm)
        for path in files
    ]
    labels = sorted(results[0].roi_values, key=roi_label_sort_key)
    expected_labels = set(labels)
    if any(set(result.roi_values) != expected_labels for result in results[1:]):
        raise ValueError("ROI label names differ between files; cannot combine them into one set of plots.")

    for x_key in ("time", "temperature", "pressure"):
        for value_index in (0, 1):
            save_plot(results, labels, x_key, value_index, output_directory)
    write_csv(results, labels, output_directory)
    print(f"Analysed {len(results)} files with ROIs: {', '.join(labels)}")
    print(f"Results written to {output_directory}")


if __name__ == "__main__":
    main()
