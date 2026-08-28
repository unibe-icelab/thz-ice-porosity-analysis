from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pydotthz import DotthzFile

from analyze_measurement_campaigns import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    get_refraction_index,
    get_thz_files,
    load_campaign_config,
    read_trace,
)


CAMPAIGN_CONFIG_PATH = DEFAULT_CONFIG_DIR / "july6_2026.json"
VACUUM_PATH = DEFAULT_DATA_DIR / "porosity_july6_2026_frost_5.0mm_5.02g/data/trans/single_pixel"
AMBIENT_PATH = (
    DEFAULT_DATA_DIR
    / "porosity_july6_2026_frost_5.0mm_5.02g_ambient_pressure/data/trans/single_pixel"
)


def find_measurement(config, sample_path: Path):
    for measurement in config.measurements:
        if measurement.path == sample_path:
            return measurement
    raise ValueError(f"Measurement not found in {CAMPAIGN_CONFIG_PATH}: {sample_path}")


def read_embedded_trace(path: Path, dataset_name: str):
    thz_files = get_thz_files(path)
    traces = []
    time_axis = None

    for thz_file in thz_files:
        with DotthzFile(thz_file) as handle:
            trace = handle["Single Pixel 0"].datasets[dataset_name][:]
            t = trace[:, 0]
            p = trace[:, 1]

        mask = (t > 1860) & (t < 1960)
        t = t[mask]
        p = p[mask]

        if time_axis is None:
            time_axis = t
        elif t.shape != time_axis.shape or not np.allclose(t, time_axis):
            p = np.interp(time_axis, t, p)

        traces.append(p)

    if time_axis is None:
        raise RuntimeError(f"No embedded {dataset_name} traces found in {path}")

    return time_axis, np.mean(np.vstack(traces), axis=0)


def estimate_time_shift(reference_time: np.ndarray, reference_trace: np.ndarray, shifted_trace: np.ndarray) -> float:
    centered_reference = reference_trace - np.mean(reference_trace)
    centered_shifted = shifted_trace - np.mean(shifted_trace)
    correlation = np.correlate(centered_reference, centered_shifted, mode="full")
    lags = np.arange(-len(centered_shifted) + 1, len(centered_reference))
    best_lag = lags[int(np.argmax(correlation))]
    sample_spacing_ps = float(np.mean(np.diff(reference_time)))
    return best_lag * sample_spacing_ps


def shift_trace(time_axis: np.ndarray, trace: np.ndarray, shift_ps: float) -> np.ndarray:
    shifted = np.interp(
        time_axis + shift_ps,
        time_axis,
        trace,
        left=np.nan,
        right=np.nan,
    )
    invalid = np.isnan(shifted)
    if np.any(invalid):
        shifted[invalid] = 0.0
    return shifted


def compute_refractive_index_curve(config, measurement, t_sample: np.ndarray, p_sample: np.ndarray):
    t_ref, p_ref, _ = read_trace(measurement.reference_path)
    frequency, refractive_index, _ = get_refraction_index(
        time=t_sample,
        traces=p_sample,
        t_ref=t_ref,
        p_ref=p_ref,
        window_half_width=config.analysis.window_half_width,
        win_func=config.analysis.window_function,
        min_frequency=config.analysis.min_frequency_thz,
        max_frequency=config.analysis.max_frequency_thz,
        d_mm=measurement.thickness_mm,
    )
    return frequency, refractive_index


def main() -> None:
    config = load_campaign_config(CAMPAIGN_CONFIG_PATH)
    vacuum_measurement = find_measurement(config, VACUUM_PATH)
    ambient_measurement = find_measurement(config, AMBIENT_PATH)

    t_vacuum, p_vacuum, _ = read_trace(vacuum_measurement.path)
    t_ambient, p_ambient_raw, _ = read_trace(ambient_measurement.path)
    t_ref_vacuum, p_ref_vacuum = read_embedded_trace(vacuum_measurement.path, "Ref")
    t_ref_ambient, p_ref_ambient = read_embedded_trace(ambient_measurement.path, "Ref")

    reference_shift_ps = estimate_time_shift(
        reference_time=t_ref_vacuum,
        reference_trace=p_ref_vacuum,
        shifted_trace=p_ref_ambient,
    )
    applied_sample_shift_ps = -reference_shift_ps
    p_ambient = shift_trace(t_ambient, p_ambient_raw, applied_sample_shift_ps)

    freq_vacuum, n_vacuum = compute_refractive_index_curve(config, vacuum_measurement, t_vacuum, p_vacuum)
    freq_ambient, n_ambient = compute_refractive_index_curve(config, ambient_measurement, t_ambient, p_ambient)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )

    fig, axes = plt.subplots(nrows=3, figsize=(8.5, 11), sharex=False)

    axes[0].plot(t_vacuum, p_vacuum, label="Vacuum", linewidth=2)
    axes[0].plot(
        t_ambient,
        p_ambient_raw,
        label="Ambient pressure (raw)",
        linewidth=1.5,
        linestyle=":",
        color="tab:gray",
    )
    axes[0].plot(t_ambient, p_ambient, label="Ambient pressure (shift-compensated)", linewidth=2, linestyle="--")
    axes[0].set_xlabel(r"$t$ [ps]")
    axes[0].set_ylabel("Amplitude [-]")
    axes[0].set_title(
        f"July 6 frost time trace: vacuum vs ambient pressure (applied shift {applied_sample_shift_ps:.3f} ps)"
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t_ref_vacuum, p_ref_vacuum, label="Vacuum Ref", linewidth=2)
    axes[1].plot(t_ref_ambient, p_ref_ambient, label="Ambient pressure Ref", linewidth=2, linestyle="--")
    axes[1].set_xlabel(r"$t$ [ps]")
    axes[1].set_ylabel("Amplitude [-]")
    axes[1].set_title("July 6 frost embedded reference pulses used for shift estimation")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(freq_vacuum, n_vacuum, label="Vacuum", linewidth=2)
    axes[2].plot(freq_ambient, n_ambient, label="Ambient pressure (shift-compensated)", linewidth=2, linestyle="--")
    axes[2].set_xlabel(r"$f$ [THz]")
    axes[2].set_ylabel(r"Refractive index $n$ [-]")
    axes[2].set_title("July 6 frost refractive index: vacuum vs ambient pressure")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    fig.tight_layout()

    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "vacuum_vs_ambient_pressure.png", dpi=300)
    fig.savefig(output_dir / "vacuum_vs_ambient_pressure.pdf")
    plt.show()


if __name__ == "__main__":
    main()
