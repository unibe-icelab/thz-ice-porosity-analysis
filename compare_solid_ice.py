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

N_LITERATURE_PATH = DEFAULT_DATA_ROOT / "ice_refractive_index.csv"
ABSORPTION_LITERATURE_PATH = DEFAULT_DATA_ROOT / "ice_absorption_coefficient.csv"


def load_literature_curve_n() -> pd.DataFrame:
    """Load the tab-separated Tao et al. frequency/index reference table."""
    df = pd.read_csv(N_LITERATURE_PATH, delimiter="\t")
    return df.rename(columns={"f": "frequency_thz", "n": "refractive_index"})

def load_literature_curve_absorption() -> pd.DataFrame:
    """Load the tab-separated Tao et al. frequency/index reference table."""
    df = pd.read_csv(ABSORPTION_LITERATURE_PATH, delimiter="\t")
    return df.rename(columns={"f": "frequency_thz", "alpha": "absorption_coefficient"})



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
    """Extract drift-corrected refractive-index and absorption spectra."""

    curves: list[dict[str, np.ndarray | str]] = []
    embedded_ref_cache: dict[Path, tuple[np.ndarray, np.ndarray]] = {}

    c = 299792458.0  # Speed of light [m/s]

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

        frequency, refractive_index, complex_n = get_refraction_index(
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

        # Extinction coefficient
        kappa = np.imag(complex_n)

        # Intensity absorption coefficient [cm^-1]
        absorption_coefficient = (
            4 * np.pi * frequency * 1e12 * kappa
            / (c * 100)
        )

        curves.append(
            {
                "label": (
                    f"{config.campaign_id} "
                    f"{measurement.path.parent.parent.name}"
                ),
                "frequency_thz": frequency,
                "refractive_index": refractive_index,
                "absorption_coefficient": absorption_coefficient,
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
    params = {
        "font.size": 20,
        "legend.fontsize": 14,
    }
    plt.rcParams.update(params)

    literature_df_n = load_literature_curve_n()
    literature_df_absorption = load_literature_curve_absorption()
    configs = load_campaign_configs()
    solid_measurements = collect_solid_measurements(configs)
    solid_curves = compute_solid_curves(solid_measurements)

    if not solid_curves:
        raise RuntimeError("No solid ice measurements found in the campaign configs.")

    frequency_thz, stacked_curves = stack_curves(solid_curves)

    # Mean refractive-index spectrum
    mean_curve = np.nanmean(stacked_curves, axis=0)

    # Mean absorption spectrum
    stacked_absorption = np.vstack([
        np.asarray(curve["absorption_coefficient"])
        for curve in solid_curves
    ])
    mean_absorption = np.nanmean(stacked_absorption, axis=0)

    lower_threshold = 0.8
    upper_threshold = 1.7

    fig, (ax_n, ax_alpha) = plt.subplots(
        2, 1,
        figsize=(16, 9),
        sharex=True,
    )

    # Literature refractive index
    lit_mask_n = (
            (literature_df_n["frequency_thz"] > lower_threshold)
            & (literature_df_n["frequency_thz"] < upper_threshold)
    )

    lit_mask_absorption = (
            (literature_df_absorption["frequency_thz"] > lower_threshold)
            & (literature_df_absorption["frequency_thz"] < upper_threshold)
    )

    ax_n.plot(
        literature_df_n["frequency_thz"][lit_mask_n],
        literature_df_n["refractive_index"][lit_mask_n],
        label="Tao et al. (2024)",
        color="tab:blue",
        linewidth=2.5,
    )

    ax_alpha.plot(
        literature_df_absorption["frequency_thz"][lit_mask_absorption],
        literature_df_absorption["absorption_coefficient"][lit_mask_absorption],
        label="Tao et al. (2024)",
        color="tab:blue",
        linewidth=2.5,
    )

    # Individual solid-ice measurements
    for i, curve in enumerate(solid_curves):
        f = np.asarray(curve["frequency_thz"])
        n = np.asarray(curve["refractive_index"])
        alpha = np.asarray(curve["absorption_coefficient"])

        mask = (f > lower_threshold) & (f < upper_threshold)

        # Refractive index
        line, = ax_n.plot(
            f[mask],
            n[mask],
            linewidth=1.2,
            linestyle="--",
            label=f"dataset {i}",
        )

        color = line.get_color()

        # Absorption coefficient (same color)
        ax_alpha.plot(
            f[mask],
            alpha[mask],
            linewidth=1.2,
            linestyle="--",
            color=color,
        )

    # Mean curves
    mean_mask = (
            (frequency_thz > lower_threshold)
            & (frequency_thz < upper_threshold)
    )

    ax_n.plot(
        frequency_thz[mean_mask],
        mean_curve[mean_mask],
        color="black",
        linewidth=2,
        label="Mean",
        zorder=3,
    )

    ax_alpha.plot(
        frequency_thz[mean_mask],
        mean_absorption[mean_mask],
        color="black",
        linewidth=2,
        label="Mean",
        zorder=3,
    )

    # Formatting
    ax_n.set_ylabel(r"Refractive index $n$ [-]")
    ax_alpha.set_ylabel(
        r"Absorption coefficient $\alpha$ [cm$^{-1}$]"
    )
    ax_alpha.set_xlabel(r"$f$ [THz]")

    ax_n.set_title("Solid ice refractive index and absorption")

    for ax in (ax_n, ax_alpha):
        ax.grid(True, alpha=0.3)

    # Legend outside, on the right
    ax_n.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )

    fig.tight_layout(rect=(0, 0, 0.95, 1))

    save_figure_outputs(
        fig,
        DEFAULT_OUTPUT_DIR / "solid_ice",
    )

    plt.show()