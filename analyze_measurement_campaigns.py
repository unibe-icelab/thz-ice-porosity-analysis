from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from pydotthz import DotthzFile
from thzpy.timedomain import common_window
from thzpy.transferfunctions import uniform_slab

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MEASUREMENT_DIR = PROJECT_ROOT / "measurements"
DEFAULT_DATA_DIR = DEFAULT_MEASUREMENT_DIR / "data"
DEFAULT_CONFIG_DIR = DEFAULT_MEASUREMENT_DIR / "configs"
DEFAULT_OUTPUT_DIR = DEFAULT_MEASUREMENT_DIR / "output"
TRACE_TIME_MIN_PS = 1860
TRACE_TIME_MAX_PS = 1960
EPS_AIR = 1.0
SOLID_ICE_DENSITY_THRESHOLD_G_CM3 = 0.8
EMT_MODELS = ("bruggemann", "maxwellgarnett", "lll")


@dataclass
class AnalysisSettings:
    radius_m: float
    radius_err_m: float
    thickness_err_mm: float
    mass_err_kg: float
    window_half_width: int
    window_function: str
    min_frequency_thz: float
    max_frequency_thz: float
    fit_band_thz: Tuple[float, float]
    solid_ice_window_half_width: int
    solid_ice_measurement_path: Path
    solid_ice_reference_path: Path
    solid_ice_thickness_mm: float
    solid_ice_density_g_cm3: float


@dataclass
class MeasurementConfig:
    measurement_id: str
    label: str
    path: Path
    reference_path: Path
    mass_kg: float
    mass_err_kg: float
    thickness_mm: float
    thickness_err_mm: float
    ignore: bool


@dataclass
class CampaignConfig:
    campaign_id: str
    description: str
    plot_color: str
    reference_path: Path
    analysis: AnalysisSettings
    measurements: List[MeasurementConfig]


@dataclass
class MeasurementResult:
    campaign_id: str
    measurement_id: str
    label: str
    density_g_cm3: float
    density_err_g_cm3: float
    refractive_index_at_1thz: float
    refractive_index_err: float
    sample_path: Path
    reference_path: Path
    sample_file_count: int
    reference_file_count: int
    complex_refractive_index_at_1thz: complex


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_project_path(value: str | Path) -> Path:
    """Resolve portable config paths relative to the repository root."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_campaign_config(path: Path) -> CampaignConfig:
    payload = load_json(path)
    analysis_payload = payload["analysis"]
    reference_payload = payload["reference"]
    solid_ice_measurement_path = analysis_payload.get("solid_ice_measurement_path")
    if solid_ice_measurement_path is None:
        solid_ice_measurement_path = analysis_payload["solid_ice_measurement_file"]

    solid_ice_reference_path = analysis_payload.get("solid_ice_reference_path")
    if solid_ice_reference_path is None:
        solid_ice_reference_path = analysis_payload["solid_ice_reference_file"]

    analysis = AnalysisSettings(
        radius_m=float(analysis_payload["radius_m"]),
        radius_err_m=float(analysis_payload["radius_err_m"]),
        thickness_err_mm=float(analysis_payload["thickness_err_mm"]),
        mass_err_kg=float(analysis_payload["mass_err_kg"]),
        window_half_width=int(analysis_payload["window_half_width"]),
        window_function=str(analysis_payload["window_function"]),
        min_frequency_thz=float(analysis_payload["min_frequency_thz"]),
        max_frequency_thz=float(analysis_payload["max_frequency_thz"]),
        fit_band_thz=(
            float(analysis_payload["fit_band_thz"][0]),
            float(analysis_payload["fit_band_thz"][1]),
        ),
        solid_ice_window_half_width=int(analysis_payload["solid_ice_window_half_width"]),
        solid_ice_measurement_path=resolve_project_path(solid_ice_measurement_path),
        solid_ice_reference_path=resolve_project_path(solid_ice_reference_path),
        solid_ice_thickness_mm=float(analysis_payload["solid_ice_thickness_mm"]),
        solid_ice_density_g_cm3=float(analysis_payload["solid_ice_density_g_cm3"]),
    )

    measurements = []
    for item in payload["measurements"]:
        measurements.append(
            MeasurementConfig(
                measurement_id=str(item["measurement_id"]),
                label=str(item["label"]),
                path=resolve_project_path(item["path"]),
                reference_path=resolve_project_path(item.get("reference_path", reference_payload["path"])),
                mass_kg=float(item["mass_kg"]),
                mass_err_kg=float(item["mass_err_kg"]),
                thickness_mm=float(item["thickness_mm"]),
                thickness_err_mm=float(item["thickness_err_mm"]),
                ignore=bool(item.get("ignore", False)),
            )
        )

    return CampaignConfig(
        campaign_id=str(payload["campaign_id"]),
        description=str(payload["description"]),
        plot_color=str(payload["plot_color"]),
        reference_path=resolve_project_path(reference_payload["path"]),
        analysis=analysis,
        measurements=measurements,
    )


def get_thz_files(path: Path) -> List[Path]:
    if path.is_file():
        if path.suffix.lower() != ".thz":
            raise ValueError(f"Expected a .thz file, got {path}")
        return [path]

    if not path.is_dir():
        raise FileNotFoundError(f"Path does not exist: {path}")

    files = sorted(candidate for candidate in path.iterdir() if candidate.suffix.lower() == ".thz")
    if not files:
        raise FileNotFoundError(f"No .thz files found in {path}")
    return files


def read_trace(path: Path) -> Tuple[np.ndarray, np.ndarray, int]:
    time_axis: np.ndarray | None = None
    traces: List[np.ndarray] = []
    thz_files = get_thz_files(path)

    for thz_file in thz_files:
        with DotthzFile(thz_file) as handle:
            sample = handle["Single Pixel 0"].datasets["Sample"][:]
            t = sample[:, 0]
            p = sample[:, 1]

        mask = (t > TRACE_TIME_MIN_PS) & (t < TRACE_TIME_MAX_PS)
        t = t[mask]
        p = p[mask]

        if time_axis is None:
            time_axis = t
            traces.append(p)
            continue

        if t.shape != time_axis.shape or not np.allclose(t, time_axis):
            p = np.interp(time_axis, t, p)
        traces.append(p)

    assert time_axis is not None
    mean_trace = np.mean(np.vstack(traces), axis=0)
    return time_axis, mean_trace, len(thz_files)


def read_trace_cached(
        path: Path,
        cache: Dict[Path, Tuple[np.ndarray, np.ndarray, int]],
) -> Tuple[np.ndarray, np.ndarray, int]:
    if path not in cache:
        cache[path] = read_trace(path)
    return cache[path]


def get_refraction_index(
        time: np.ndarray,
        traces: np.ndarray,
        t_ref: np.ndarray,
        p_ref: np.ndarray,
        window_half_width: int,
        win_func: str,
        min_frequency: float,
        max_frequency: float,
        d_mm: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_ref = np.array([t_ref, p_ref])
    p_pair = np.array([time, traces])

    sample, reference = common_window(
        [p_pair, data_ref],
        half_width=window_half_width,
        win_func=win_func,
    )
    response = uniform_slab(
        d_mm,
        sample,
        reference,
        n_med=1,
        upsampling=1,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
        all_optical_constants=True,
    )

    frequency = np.array(response["frequency"])
    complex_refractive_index = np.array(response["refractive_index"]).astype(complex)
    refractive_index = complex_refractive_index.real
    return frequency, refractive_index, complex_refractive_index


def compute_density(mass_kg: float, thickness_mm: float, settings: AnalysisSettings) -> Tuple[float, float]:
    thickness_m = thickness_mm / 1000.0
    thickness_err_m = settings.thickness_err_mm / 1000.0
    radius_m = settings.radius_m
    radius_err_m = settings.radius_err_m
    volume_m3 = np.pi * radius_m ** 2 * thickness_m
    density_kg_m3 = mass_kg / volume_m3

    density_err_kg_m3 = np.sqrt(
        (settings.mass_err_kg / volume_m3) ** 2
        + (mass_kg / (np.pi * radius_m ** 2 * thickness_m ** 2) * thickness_err_m) ** 2
        + (2 * mass_kg / (np.pi * radius_m ** 3 * thickness_m) * radius_err_m) ** 2
    )
    return density_kg_m3 / 1000.0, density_err_kg_m3 / 1000.0


def compute_refractive_index_error(refractive_index_at_1thz: float, thickness_mm: float,
                                   thickness_err_mm: float) -> float:
    if thickness_mm == 0:
        return float("nan")
    return refractive_index_at_1thz / thickness_mm * thickness_err_mm


def maxwell_garnett_eps(volume_fraction_ice: np.ndarray, eps_host: complex, eps_ice: complex) -> np.ndarray:
    delta_eps = eps_ice - eps_host
    numerator = eps_ice + 2 * eps_host + 2 * volume_fraction_ice * delta_eps
    denominator = eps_ice + 2 * eps_host - volume_fraction_ice * delta_eps
    return eps_host * numerator / denominator


def lll_eps(volume_fraction_ice: np.ndarray, eps_host: complex, eps_ice: complex) -> np.ndarray:
    eps_host_root = np.power(eps_host, 1 / 3)
    eps_ice_root = np.power(eps_ice, 1 / 3)
    return np.power((1 - volume_fraction_ice) * eps_host_root + volume_fraction_ice * eps_ice_root, 3)


def bruggemann_eps(volume_fraction_ice: np.ndarray, eps_host: complex, eps_ice: complex) -> np.ndarray:
    b_term = 3 * volume_fraction_ice * (eps_ice - eps_host) + 2 * eps_host - eps_ice
    return (b_term + np.sqrt(b_term ** 2 + 8 * eps_host * eps_ice)) / 4


def theoretical_emt_curve(
        solid_ice_complex_index: complex,
        solid_ice_density_g_cm3: float,
        model: str,
        point_count: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    density_axis = np.linspace(0.0, solid_ice_density_g_cm3, point_count)
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


def emt_model_label(model: str) -> str:
    if model == "bruggemann":
        return "Bruggemann"
    if model == "maxwellgarnett":
        return "Maxwell-Garnett"
    if model == "lll":
        return "LLL"
    raise ValueError(f"Unsupported EMT model: {model}")


def analyze_campaign(
        campaign: CampaignConfig,
        overview_ax: Sequence[plt.Axes],
        scatter_ax: plt.Axes,
        trace_cache: Dict[Path, Tuple[np.ndarray, np.ndarray, int]],
) -> List[MeasurementResult]:
    settings = campaign.analysis
    scatter_color = campaign.plot_color
    fit_band_min, fit_band_max = settings.fit_band_thz

    results: List[MeasurementResult] = []
    for measurement in campaign.measurements:
        if measurement.ignore:
            continue

        sample_t, sample_p, sample_file_count = read_trace_cached(measurement.path, trace_cache)
        reference_t, reference_p, reference_file_count = read_trace_cached(measurement.reference_path, trace_cache)
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

        overview_ax[0].plot(sample_t, sample_p, label=measurement.measurement_id)
        overview_ax[1].plot(freqs, refractive_index, label=measurement.measurement_id)
        scatter_ax.errorbar(
            density_g_cm3,
            refractive_index_at_1thz,
            xerr=density_err_g_cm3,
            yerr=refractive_index_err,
            fmt="o",
            color=scatter_color,
            ecolor="black",
            markersize=5,
            capsize=2,
            markeredgecolor="black",
            markeredgewidth=0.5,
            label=campaign.campaign_id,
        )

        results.append(
            MeasurementResult(
                campaign_id=campaign.campaign_id,
                measurement_id=measurement.measurement_id,
                label=measurement.label,
                density_g_cm3=density_g_cm3,
                density_err_g_cm3=density_err_g_cm3,
                refractive_index_at_1thz=refractive_index_at_1thz,
                refractive_index_err=refractive_index_err,
                sample_path=measurement.path,
                reference_path=measurement.reference_path,
                sample_file_count=sample_file_count,
                reference_file_count=reference_file_count,
                complex_refractive_index_at_1thz=complex_refractive_index_at_1thz,
            )
        )

    return results


def plot_shared_emt_curves(scatter_ax: plt.Axes, results: Sequence[MeasurementResult]) -> Tuple[complex, float, int]:
    solid_results = [
        result for result in results
        if result.density_g_cm3 > SOLID_ICE_DENSITY_THRESHOLD_G_CM3
    ]
    if not solid_results:
        raise ValueError(
            f"No measurements above {SOLID_ICE_DENSITY_THRESHOLD_G_CM3:.1f} g/cm^3 available for shared EMT calibration."
        )

    mean_complex_index = complex(np.mean([result.complex_refractive_index_at_1thz for result in solid_results]))
    mean_density_g_cm3 = float(np.mean([result.density_g_cm3 for result in solid_results]))

    model_styles = {
        "bruggemann": {"color": "black", "linestyle": "--"},
        "maxwellgarnett": {"color": "dimgray", "linestyle": "-."},
        "lll": {"color": "saddlebrown", "linestyle": ":"},
    }
    for model in EMT_MODELS:
        emt_density_axis, emt_refractive_index = theoretical_emt_curve(
            solid_ice_complex_index=mean_complex_index,
            solid_ice_density_g_cm3=mean_density_g_cm3,
            model=model,
        )
        style = model_styles[model]
        scatter_ax.plot(
            emt_density_axis,
            emt_refractive_index,
            linewidth=1.8,
            label=f"{emt_model_label(model)} EMT",
            **style,
        )

    return mean_complex_index, mean_density_g_cm3, len(solid_results)


def write_summary_csv(results: Iterable[MeasurementResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "campaign_id",
                "measurement_id",
                "label",
                "density_g_cm3",
                "density_err_g_cm3",
                "refractive_index_at_1thz",
                "refractive_index_err",
                "sample_path",
                "reference_path",
                "sample_file_count",
                "reference_file_count",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.campaign_id,
                    result.measurement_id,
                    result.label,
                    result.density_g_cm3,
                    result.density_err_g_cm3,
                    result.refractive_index_at_1thz,
                    result.refractive_index_err,
                    str(result.sample_path),
                    str(result.reference_path),
                    result.sample_file_count,
                    result.reference_file_count,
                ]
            )


def collect_config_paths(config_dir: Path, config_files: Sequence[str]) -> List[Path]:
    if config_files:
        return [Path(item).resolve() for item in config_files]
    return sorted(config_dir.glob("*.json"))


def make_unique_legend(ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    unique: Dict[str, object] = {}
    for handle, label in zip(handles, labels):
        if label not in unique:
            unique[label] = handle
    ax.legend(unique.values(), unique.keys(), fontsize=8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze THz measurement campaigns from JSON configs.")
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
        help="Directory for CSV and plot outputs.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively in addition to saving them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_paths = collect_config_paths(args.config_dir.resolve(), args.config)
    if not config_paths:
        raise FileNotFoundError("No campaign config files found.")

    campaigns = [load_campaign_config(path) for path in config_paths]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scatter_fig, scatter_ax = plt.subplots(figsize=(8, 6))
    overview_figures: List[plt.Figure] = []
    trace_cache: Dict[Path, Tuple[np.ndarray, np.ndarray, int]] = {}

    all_results: List[MeasurementResult] = []
    for campaign in campaigns:
        overview_fig, overview_axes = plt.subplots(nrows=2, figsize=(10, 8))
        all_results.extend(analyze_campaign(campaign, overview_axes, scatter_ax, trace_cache))

        overview_axes[0].set_title(campaign.campaign_id)
        overview_axes[0].set_xlabel("Time (ps)")
        overview_axes[0].set_ylabel("Amplitude (a.u.)")
        overview_axes[1].set_xlabel("Frequency (THz)")
        overview_axes[1].set_ylabel("Refractive index (-)")
        make_unique_legend(overview_axes[0])
        overview_fig.tight_layout()
        overview_fig.savefig(args.output_dir / f"{campaign.campaign_id}_trace_overview.png", dpi=300)
        overview_figures.append(overview_fig)

    _, mean_density_g_cm3, solid_result_count = plot_shared_emt_curves(scatter_ax, all_results)
    scatter_ax.set_xlabel(r"Ice density [g/cm$^3$]")
    scatter_ax.set_ylabel("Refractive index @ 1 THz (-)")
    scatter_ax.set_title(
        f"Measured points with shared EMT curves ({solid_result_count} solids, avg density {mean_density_g_cm3:.3f} g/cm^3)"
    )
    make_unique_legend(scatter_ax)

    write_summary_csv(all_results, args.output_dir / "campaign_analysis_summary.csv")

    scatter_fig.tight_layout()
    scatter_fig.savefig(args.output_dir / "campaign_refractive_index_vs_density.png", dpi=300)

    if args.show:
        plt.show()
    else:
        for overview_fig in overview_figures:
            plt.close(overview_fig)
        plt.close(scatter_fig)


if __name__ == "__main__":
    main()
