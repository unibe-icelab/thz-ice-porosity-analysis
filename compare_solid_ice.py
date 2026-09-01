"""Compare drift-corrected solid-ice THz spectra with Tao et al. data."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_measurement_campaigns import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_DIR,
    CampaignConfig,
    MeasurementConfig,
    get_refraction_index,
    load_campaign_config,
    read_trace,
)
from analyze_porosity_emt import compensate_sample_trace_from_embedded_refs
from plot_outputs import save_figure_outputs

LITERATURE_PATH = DEFAULT_DATA_ROOT / "ice_refractive_index.csv"


def load_literature_curve() -> pd.DataFrame:
    """Load the tab-separated Tao et al. frequency/index reference table."""
    df = pd.read_csv(LITERATURE_PATH, delimiter="\t")
    return df.rename(columns={"f": "frequency_thz", "n": "refractive_index"})


def load_campaign_configs() -> list[CampaignConfig]:
    """Load every campaign JSON file from the default configuration directory."""
    return [load_campaign_config(path) for path in sorted(DEFAULT_CONFIG_DIR.glob("*.json"))]


def collect_solid_measurements(configs: list[CampaignConfig]) -> list[tuple[CampaignConfig, MeasurementConfig]]:
    """Collect unique, non-ignored measurements explicitly labelled ``SOLID``."""
    solid_measurements: list[tuple[CampaignConfig, MeasurementConfig]] = []
    seen_paths: set[Path] = set()

    for config in configs:
        for measurement in config.measurements:
            if measurement.ignore or measurement.label != "SOLID":
                continue
            if measurement.path in seen_paths:
                continue
            seen_paths.add(measurement.path)
            solid_measurements.append((config, measurement))

    return solid_measurements


def compute_solid_curves(
        solid_measurements: list[tuple[CampaignConfig, MeasurementConfig]],
) -> list[dict[str, np.ndarray | str]]:
    """Extract drift-corrected refractive-index spectra for solid samples.

    Embedded sample/reference traces are used for timing compensation before
    the slab inversion. Each result dictionary contains a label, frequency axis
    in terahertz, and the real refractive-index spectrum.
    """
    curves: list[dict[str, np.ndarray | str]] = []
    embedded_ref_cache: dict[Path, tuple[np.ndarray, np.ndarray]] = {}

    for config, measurement in solid_measurements:
        time, sample_trace, _ = read_trace(measurement.path)
        sample_trace = compensate_sample_trace_from_embedded_refs(
            measurement_path=measurement.path,
            reference_path=measurement.reference_path,
            sample_time=time,
            sample_trace=sample_trace,
            embedded_ref_cache=embedded_ref_cache,
        )
        t_ref, p_ref, _ = read_trace(measurement.reference_path)
        frequency, refractive_index, _ = get_refraction_index(
            time=time,
            traces=sample_trace,
            t_ref=t_ref,
            p_ref=p_ref,
            window_half_width=config.analysis.solid_ice_window_half_width,
            win_func=config.analysis.window_function,
            min_frequency=config.analysis.min_frequency_thz,
            max_frequency=config.analysis.max_frequency_thz,
            d_mm=measurement.thickness_mm,
        )
        curves.append(
            {
                "label": f"{config.campaign_id} {measurement.path.parent.parent.name}",
                "frequency_thz": frequency,
                "refractive_index": refractive_index,
            }
        )

    return curves


def stack_curves(curves: list[dict[str, np.ndarray | str]]) -> tuple[np.ndarray, np.ndarray]:
    """Stack spectra that share the first curve's frequency axis."""
    frequency = np.asarray(curves[0]["frequency_thz"])
    stacked = np.vstack([np.asarray(curve["refractive_index"]) for curve in curves])
    return frequency, stacked


if __name__ == "__main__":
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )

    literature_df = load_literature_curve()
    configs = load_campaign_configs()
    solid_measurements = collect_solid_measurements(configs)
    solid_curves = compute_solid_curves(solid_measurements)

    if not solid_curves:
        raise RuntimeError("No solid ice measurements found in the campaign configs.")

    frequency_thz, stacked_curves = stack_curves(solid_curves)
    mean_curve = np.nanmean(stacked_curves, axis=0)
    lower_threshold = 0.8
    upper_threshold = 2.0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        literature_df["frequency_thz"][
            (literature_df["frequency_thz"] > lower_threshold) & (literature_df["frequency_thz"] < upper_threshold)],
        literature_df["refractive_index"][
            (literature_df["frequency_thz"] > lower_threshold) & (literature_df["frequency_thz"] < upper_threshold)],
        label="Tao et al. (2024)",
        color="tab:blue",
        linewidth=2.5,
    )

    for i, curve in enumerate(solid_curves):
        ax.plot(
            curve["frequency_thz"][
                (curve["frequency_thz"] > lower_threshold) & (curve["frequency_thz"] < upper_threshold)],
            curve["refractive_index"][
                (curve["frequency_thz"] > lower_threshold) & (curve["frequency_thz"] < upper_threshold)],
            alpha=1.0,
            linewidth=1.2,
            linestyle="--",
            label=f"dataset {i}",
        )

    ax.plot(
        frequency_thz[(frequency_thz > lower_threshold) & (frequency_thz < upper_threshold)],
        mean_curve[(frequency_thz > lower_threshold) & (frequency_thz < upper_threshold)],
        color="black",
        linewidth=2,
        label=f"Mean",
        zorder=3,
    )

    ax.set_xlabel(r"$f$ [THz]")
    ax.set_ylabel(r"Refractive index $n$ [-]")
    ax.set_title("Solid ice refractive-index")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout(rect=(0, 0, 0.95, 1))
    save_figure_outputs(fig, DEFAULT_OUTPUT_DIR / "solid_ice")
    plt.show()
