from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from pydotthz import DotthzFile

from analyze_measurement_campaigns import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_OUTPUT_DIR,
    EPS_AIR,
    SOLID_ICE_DENSITY_THRESHOLD_G_CM3,
    CampaignConfig,
    collect_config_paths,
    compute_density,
    compute_refractive_index_error,
    get_thz_files,
    get_refraction_index,
    load_campaign_config,
    make_unique_legend,
    read_trace_cached,
)


EMT_MODELS = ("bruggemann",)


@dataclass
class OpticalMeasurement:
    campaign_id: str
    label: str
    measurement_id: str
    plot_color: str
    density_g_cm3: float
    density_err_g_cm3: float
    true_porosity: float
    true_porosity_err: float
    frequencies_thz: np.ndarray
    complex_refractive_index: np.ndarray
    refractive_index_at_1thz: float
    refractive_index_err: float
    complex_refractive_index_at_1thz: complex


@dataclass
class PorosityEstimate:
    model: str
    estimated_porosity: float
    estimated_porosity_err: float
    porosity_error: float


def read_embedded_ref_trace(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    time_axis: np.ndarray | None = None
    traces: List[np.ndarray] = []

    for thz_file in get_thz_files(path):
        with DotthzFile(thz_file) as handle:
            ref_dataset = handle["Single Pixel 0"].datasets.get("Ref")
            if ref_dataset is None:
                raise KeyError(f"No embedded Ref dataset in {thz_file}")
            ref_trace = ref_dataset[:]
            t = np.asarray(ref_trace[:, 0])
            p = np.asarray(ref_trace[:, 1])

        mask = (t > 1860) & (t < 1960)
        t = t[mask]
        p = p[mask]

        if time_axis is None:
            time_axis = t
        elif t.shape != time_axis.shape or not np.allclose(t, time_axis):
            p = np.interp(time_axis, t, p)

        traces.append(p)

    if time_axis is None:
        raise RuntimeError(f"No embedded Ref traces found in {path}")

    return time_axis, np.mean(np.vstack(traces), axis=0)


def read_embedded_ref_trace_cached(
    path: Path,
    cache: Dict[Path, Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    if path not in cache:
        cache[path] = read_embedded_ref_trace(path)
    return cache[path]


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


def compensate_sample_trace_from_embedded_refs(
    measurement_path: Path,
    reference_path: Path,
    sample_time: np.ndarray,
    sample_trace: np.ndarray,
    embedded_ref_cache: Dict[Path, Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    try:
        reference_ref_t, reference_ref_p = read_embedded_ref_trace_cached(reference_path, embedded_ref_cache)
        measurement_ref_t, measurement_ref_p = read_embedded_ref_trace_cached(measurement_path, embedded_ref_cache)
    except (KeyError, RuntimeError):
        return sample_trace

    if (
        measurement_ref_t.shape != reference_ref_t.shape
        or not np.allclose(measurement_ref_t, reference_ref_t)
    ):
        measurement_ref_p = np.interp(reference_ref_t, measurement_ref_t, measurement_ref_p)
        measurement_ref_t = reference_ref_t

    reference_shift_ps = estimate_time_shift(
        reference_time=reference_ref_t,
        reference_trace=reference_ref_p,
        shifted_trace=measurement_ref_p,
    )
    applied_sample_shift_ps = -reference_shift_ps
    return shift_trace(sample_time, sample_trace, applied_sample_shift_ps)


def bruggemann_vi(epsilon_eff: complex, epsilon_ice: complex) -> complex:
    lhs = (EPS_AIR - epsilon_eff) / (EPS_AIR + 2 * epsilon_eff)
    rhs = (epsilon_ice - epsilon_eff) / (epsilon_ice + 2 * epsilon_eff)
    return lhs / (lhs - rhs)


def maxwell_garnett_vi(epsilon_eff: complex, epsilon_host: complex, epsilon_ice: complex) -> complex:
    numerator = (epsilon_eff - epsilon_host) * (epsilon_ice + 2 * epsilon_host)
    denominator = (epsilon_ice - epsilon_host) * (epsilon_eff + 2 * epsilon_host)
    return numerator / denominator


def lll_vi(epsilon_eff: complex, epsilon_host: complex, epsilon_ice: complex) -> complex:
    return (np.power(epsilon_eff, 1 / 3) - np.power(epsilon_host, 1 / 3)) / (
        np.power(epsilon_ice, 1 / 3) - np.power(epsilon_host, 1 / 3)
    )


def bruggemann_eps(volume_fraction_ice: np.ndarray, eps_host: complex, eps_ice: complex) -> np.ndarray:
    b_term = 3 * volume_fraction_ice * (eps_ice - eps_host) + 2 * eps_host - eps_ice
    return (b_term + np.sqrt(b_term ** 2 + 8 * eps_host * eps_ice)) / 4


def maxwell_garnett_eps(volume_fraction_ice: np.ndarray, eps_host: complex, eps_ice: complex) -> np.ndarray:
    delta_eps = eps_ice - eps_host
    numerator = eps_ice + 2 * eps_host + 2 * volume_fraction_ice * delta_eps
    denominator = eps_ice + 2 * eps_host - volume_fraction_ice * delta_eps
    return eps_host * numerator / denominator


def lll_eps(volume_fraction_ice: np.ndarray, eps_host: complex, eps_ice: complex) -> np.ndarray:
    eps_host_root = np.power(eps_host, 1 / 3)
    eps_ice_root = np.power(eps_ice, 1 / 3)
    return np.power((1 - volume_fraction_ice) * eps_host_root + volume_fraction_ice * eps_ice_root, 3)


def emt_model_label(model: str) -> str:
    if model == "bruggemann":
        return "Bruggemann"
    if model == "maxwellgarnett":
        return "Maxwell-Garnett"
    if model == "lll":
        return "LLL"
    raise ValueError(f"Unsupported EMT model: {model}")


def invert_volume_fraction(model: str, epsilon_eff: complex, epsilon_ice: complex) -> float:
    if model == "bruggemann":
        value = bruggemann_vi(epsilon_eff, epsilon_ice)
    elif model == "maxwellgarnett":
        value = maxwell_garnett_vi(epsilon_eff, EPS_AIR, epsilon_ice)
    elif model == "lll":
        value = lll_vi(epsilon_eff, EPS_AIR, epsilon_ice)
    else:
        raise ValueError(f"Unsupported EMT model: {model}")
    return float(np.clip(np.real(value), 0.0, 1.0))


def forward_emt_curve(model: str, solid_ice_complex_index: complex, solid_ice_density_g_cm3: float) -> Tuple[np.ndarray, np.ndarray]:
    density_axis = np.linspace(0.0, solid_ice_density_g_cm3, 300)
    volume_fraction_ice = np.clip(density_axis / solid_ice_density_g_cm3, 0.0, 1.0)
    eps_ice = solid_ice_complex_index ** 2

    if model == "bruggemann":
        eps_eff = bruggemann_eps(volume_fraction_ice, EPS_AIR, eps_ice)
    elif model == "maxwellgarnett":
        eps_eff = maxwell_garnett_eps(volume_fraction_ice, EPS_AIR, eps_ice)
    elif model == "lll":
        eps_eff = lll_eps(volume_fraction_ice, EPS_AIR, eps_ice)
    else:
        raise ValueError(f"Unsupported EMT model: {model}")

    return density_axis, np.real(np.sqrt(eps_eff))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze porosity extraction from THz-TDS using Bruggemann, Maxwell-Garnett, and LLL EMT."
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help="Directory containing campaign JSON config files.",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Specific config file(s) to analyze. Defaults to all JSON files in --config-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for plot and CSV outputs.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively in addition to saving them.",
    )
    return parser.parse_args()


def collect_measurements(campaigns: Sequence[CampaignConfig]) -> List[OpticalMeasurement]:
    trace_cache: Dict[Path, Tuple[np.ndarray, np.ndarray, int]] = {}
    embedded_ref_cache: Dict[Path, Tuple[np.ndarray, np.ndarray]] = {}
    results: List[OpticalMeasurement] = []

    for campaign in campaigns:
        settings = campaign.analysis
        fit_band_min, fit_band_max = settings.fit_band_thz

        for measurement in campaign.measurements:
            if measurement.ignore:
                continue

            sample_t, sample_p, _ = read_trace_cached(measurement.path, trace_cache)
            sample_p = compensate_sample_trace_from_embedded_refs(
                measurement_path=measurement.path,
                reference_path=measurement.reference_path,
                sample_time=sample_t,
                sample_trace=sample_p,
                embedded_ref_cache=embedded_ref_cache,
            )
            reference_t, reference_p, _ = read_trace_cached(measurement.reference_path, trace_cache)
            freqs, refractive_index, complex_refractive_index = get_refraction_index(
                sample_t,
                sample_p,
                reference_t,
                reference_p,
                window_half_width=settings.window_half_width,
                win_func=settings.window_function,
                min_frequency=settings.min_frequency_thz,
                max_frequency=settings.max_frequency_thz,
                d_mm=measurement.thickness_mm,
            )

            density_g_cm3, density_err_g_cm3 = compute_density(
                measurement.mass_kg,
                measurement.thickness_mm,
                settings,
            )
            fit_mask = (freqs > fit_band_min) & (freqs < fit_band_max)
            refractive_index_at_1thz = float(np.nanmean(refractive_index[fit_mask]))
            complex_refractive_index_at_1thz = complex(np.nanmean(complex_refractive_index[fit_mask]))
            refractive_index_err = compute_refractive_index_error(
                refractive_index_at_1thz,
                measurement.thickness_mm,
                measurement.thickness_err_mm,
            )

            results.append(
                OpticalMeasurement(
                    campaign_id=campaign.campaign_id,
                    label=measurement.label,
                    measurement_id=measurement.measurement_id,
                    plot_color=campaign.plot_color,
                    density_g_cm3=density_g_cm3,
                    density_err_g_cm3=density_err_g_cm3,
                    true_porosity=float("nan"),
                    true_porosity_err=float("nan"),
                    frequencies_thz=freqs,
                    complex_refractive_index=complex_refractive_index,
                    refractive_index_at_1thz=refractive_index_at_1thz,
                    refractive_index_err=refractive_index_err,
                    complex_refractive_index_at_1thz=complex_refractive_index_at_1thz,
                )
            )

    return results


def calibrate_shared_solid_ice(measurements: List[OpticalMeasurement]) -> Tuple[complex, float, float]:
    solid_measurements = [
        measurement for measurement in measurements
        if measurement.density_g_cm3 > SOLID_ICE_DENSITY_THRESHOLD_G_CM3
    ]
    if not solid_measurements:
        raise ValueError(
            f"No measurements above {SOLID_ICE_DENSITY_THRESHOLD_G_CM3:.1f} g/cm^3 available for EMT calibration."
        )

    mean_complex_index = complex(np.mean([measurement.complex_refractive_index_at_1thz for measurement in solid_measurements]))
    mean_density_g_cm3 = float(np.mean([measurement.density_g_cm3 for measurement in solid_measurements]))
    if len(solid_measurements) > 1:
        density_std_g_cm3 = float(np.std([measurement.density_g_cm3 for measurement in solid_measurements], ddof=1))
    else:
        density_std_g_cm3 = 0.0

    for measurement in measurements:
        measurement.true_porosity = 1.0 - measurement.density_g_cm3 / mean_density_g_cm3
        measurement.true_porosity_err = np.sqrt(
            (measurement.density_err_g_cm3 / mean_density_g_cm3) ** 2
            + (measurement.density_g_cm3 * density_std_g_cm3 / (mean_density_g_cm3 ** 2)) ** 2
        )

    return mean_complex_index, mean_density_g_cm3, density_std_g_cm3


def estimate_porosity(
    measurement: OpticalMeasurement,
    model: str,
    epsilon_ice: complex,
) -> PorosityEstimate:
    epsilon_eff = measurement.complex_refractive_index_at_1thz ** 2
    volume_fraction_ice = invert_volume_fraction(model, epsilon_eff, epsilon_ice)
    estimated_porosity = 1.0 - volume_fraction_ice

    n_lower = complex(
        measurement.refractive_index_at_1thz - measurement.refractive_index_err,
        measurement.complex_refractive_index_at_1thz.imag,
    )
    n_upper = complex(
        measurement.refractive_index_at_1thz + measurement.refractive_index_err,
        measurement.complex_refractive_index_at_1thz.imag,
    )
    porosity_lower = 1.0 - invert_volume_fraction(model, n_lower ** 2, epsilon_ice)
    porosity_upper = 1.0 - invert_volume_fraction(model, n_upper ** 2, epsilon_ice)
    estimated_porosity_err = 0.5 * abs(porosity_upper - porosity_lower)

    return PorosityEstimate(
        model=model,
        estimated_porosity=estimated_porosity,
        estimated_porosity_err=estimated_porosity_err,
        porosity_error=estimated_porosity - measurement.true_porosity,
    )


def plot_summary_figure(
    measurements: Sequence[OpticalMeasurement],
    solid_ice_complex_index: complex,
    solid_ice_density_g_cm3: float,
    bruggemann_estimates: Sequence[PorosityEstimate],
    output_path: Path,
) -> plt.Figure:
    fig, axes = plt.subplots(ncols=2, figsize=(13, 5), constrained_layout=True)
    ax_forward, ax_residual = axes
    for measurement in measurements:
        ax_forward.errorbar(
            measurement.density_g_cm3,
            measurement.refractive_index_at_1thz,
            xerr=measurement.density_err_g_cm3,
            yerr=measurement.refractive_index_err,
            fmt="o",
            color=measurement.plot_color,
            ecolor="black",
            markersize=5,
            capsize=2,
            markeredgecolor="black",
            markeredgewidth=0.5,
            label=measurement.campaign_id,
        )

    density_axis, refractive_index_curve = forward_emt_curve(
        "bruggemann",
        solid_ice_complex_index,
        solid_ice_density_g_cm3,
    )
    ax_forward.plot(
        density_axis,
        refractive_index_curve,
        linewidth=1.8,
        color="black",
        linestyle="--",
        label="Bruggemann EMT",
    )
    ax_forward.set_xlabel(r"Ice density [g/cm$^3$]")
    ax_forward.set_ylabel("Refractive index @ 1 THz [-]")
    ax_forward.set_title("a) Refractive Index")
    make_unique_legend(ax_forward)

    ax_residual.axhline(0.0, color="black", linewidth=1.0)
    for measurement, estimate in zip(measurements, bruggemann_estimates):
        ax_residual.errorbar(
            measurement.true_porosity,
            estimate.porosity_error,
            xerr=measurement.true_porosity_err,
            yerr=estimate.estimated_porosity_err,
            fmt="o",
            color=measurement.plot_color,
            ecolor="black",
            markersize=5,
            capsize=2,
            markeredgecolor="black",
            markeredgewidth=0.5,
            label=measurement.campaign_id,
        )
    ax_residual.set_xlabel("True porosity [-]")
    ax_residual.set_ylabel("Estimated porosity - true porosity [-]")
    ax_residual.set_title("b) Bruggemann Residuals")
    make_unique_legend(ax_residual)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    return fig


def write_measurement_summary(
    measurements: Sequence[OpticalMeasurement],
    estimates_by_model: Dict[str, List[PorosityEstimate]],
    output_path: Path,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "campaign_id",
                "measurement_id",
                "label",
                "density_g_cm3",
                "density_err_g_cm3",
                "true_porosity",
                "true_porosity_err",
                "refractive_index_at_1thz",
                "refractive_index_err",
                "bruggemann_porosity",
                "bruggemann_porosity_err",
                "bruggemann_porosity_error",
            ]
        )
        for index, measurement in enumerate(measurements):
            writer.writerow(
                [
                    measurement.campaign_id,
                    measurement.measurement_id,
                    measurement.label,
                    measurement.density_g_cm3,
                    measurement.density_err_g_cm3,
                    measurement.true_porosity,
                    measurement.true_porosity_err,
                    measurement.refractive_index_at_1thz,
                    measurement.refractive_index_err,
                    estimates_by_model["bruggemann"][index].estimated_porosity,
                    estimates_by_model["bruggemann"][index].estimated_porosity_err,
                    estimates_by_model["bruggemann"][index].porosity_error,
                ]
            )


def write_model_metrics(
    estimates_by_model: Dict[str, List[PorosityEstimate]],
    output_path: Path,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "mae", "rmse", "bias", "max_abs_error"])
        for model in EMT_MODELS:
            errors = np.array([estimate.porosity_error for estimate in estimates_by_model[model]], dtype=float)
            writer.writerow(
                [
                    emt_model_label(model),
                    float(np.mean(np.abs(errors))),
                    float(np.sqrt(np.mean(errors ** 2))),
                    float(np.mean(errors)),
                    float(np.max(np.abs(errors))),
                ]
            )


def main() -> None:
    args = parse_args()
    config_paths = collect_config_paths(args.config_dir.resolve(), args.config)
    if not config_paths:
        raise FileNotFoundError("No campaign config files found.")

    campaigns = [load_campaign_config(path) for path in config_paths]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    measurements = collect_measurements(campaigns)
    solid_ice_complex_index, solid_ice_density_g_cm3, _ = calibrate_shared_solid_ice(measurements)
    epsilon_ice = solid_ice_complex_index ** 2

    estimates_by_model: Dict[str, List[PorosityEstimate]] = {model: [] for model in EMT_MODELS}
    for measurement in measurements:
        for model in EMT_MODELS:
            estimates_by_model[model].append(estimate_porosity(measurement, model, epsilon_ice))

    figures = [
        plot_summary_figure(
        measurements,
        solid_ice_complex_index,
        solid_ice_density_g_cm3,
        estimates_by_model["bruggemann"],
        args.output_dir / "emt_bruggemann_summary.pdf",
        ),
    ]
    write_measurement_summary(
        measurements,
        estimates_by_model,
        args.output_dir / "emt_porosity_measurements.csv",
    )
    write_model_metrics(
        estimates_by_model,
        args.output_dir / "emt_porosity_model_metrics.csv",
    )

    plt.show()



params = {"ytick.color":
              "black",

          "xtick.color":
              "black",

          "axes.labelcolor":
              "black",

          "axes.edgecolor":
              "black",

          "mathtext.fontset":
              "cm",

          "mathtext.rm":
              "Times New Roman",

          "font.family":
              "Helvetica Neue",

          "font.size":
              18}

plt.rcParams.update(params)

if __name__ == "__main__":
    main()
