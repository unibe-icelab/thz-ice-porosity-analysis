"""Shared THz-TDS campaign loading, optical extraction, and EMT utilities.

The module reads campaign JSON files and ``.thz`` acquisitions, averages
repeat traces, derives slab optical constants, propagates geometry
uncertainties, and constructs effective-medium curves at 1 THz. Paths are
resolved relative to the repository so the scripts can run from any directory.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from pydotthz import DotthzFile
from thzpy.timedomain import common_window
from thzpy.transferfunctions import uniform_slab

from plot_outputs import save_figure_outputs

params = {
    "font.size": 20,
    "legend.fontsize": 14,
}
plt.rcParams.update(params)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_DATA_DIR = DEFAULT_DATA_ROOT / "raw"
DEFAULT_CONFIG_DIR = DEFAULT_DATA_ROOT / "configs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results"
TAO_ICE_REFRACTIVE_INDEX_PATH = DEFAULT_DATA_ROOT / "ice_refractive_index.csv"
TRACE_TIME_MIN_PS = 1860
TRACE_TIME_MAX_PS = 1960
EPS_VACUUM = 1.0
SOLID_ICE_DENSITY_G_CM3 = 0.917
EMT_MODELS = ("bruggeman",)


@dataclass
class AnalysisSettings:
    """Analysis parameters shared by all measurements in one campaign.

    Lengths retain the units encoded in each attribute name. Uncertainty fields
    are interpreted as standard uncertainties.
    """
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
    """Configuration and physical metadata for one sample acquisition."""
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
    """Fully resolved campaign configuration with its measurement records."""
    campaign_id: str
    description: str
    plot_color: str
    reference_path: Path
    analysis: AnalysisSettings
    measurements: List[MeasurementConfig]


@dataclass
class MeasurementResult:
    """Derived density and 1 THz optical result for one measurement."""
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
    """Read and decode a UTF-8 JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_project_path(value: str | Path) -> Path:
    """Resolve portable config paths relative to the repository root."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_campaign_config(path: Path) -> CampaignConfig:
    """Load a campaign JSON file and resolve all data paths.

    Both current ``*_path`` and legacy ``*_file`` solid-reference keys are
    accepted. Numeric JSON values are normalized to their declared Python
    types before the dataclasses are constructed.
    """
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
    """Return sorted THz acquisition files from a file or directory path.

    Raises:
        ValueError: If a supplied file does not have the ``.thz`` suffix.
        FileNotFoundError: If the path or matching acquisition files are absent.
    """
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
    """Read, crop, align, and average sample traces from an acquisition path.

    Returns:
        A tuple ``(time_ps, mean_amplitude, file_count)``. Traces whose time
        axes differ are linearly interpolated onto the first trace's axis.
    """
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
    """Return an averaged trace, populating the caller-owned cache on demand."""
    if path not in cache:
        cache[path] = read_trace(path)
    return cache[path]


def compute_time_domain_snr(
        time: np.ndarray,
        trace: np.ndarray,
        noise_fraction: float = 0.1,
) -> float:
    """Estimate peak-to-RMS SNR in dB using the first part of a trace."""
    time = np.asarray(time)
    trace = np.asarray(trace)

    # Use the first 10% of the trace as the noise region
    n_noise = max(2, int(len(trace) * noise_fraction))
    noise_region = trace[:n_noise]

    noise_rms = np.std(noise_region)
    signal_peak = np.max(np.abs(trace - np.mean(noise_region)))

    if noise_rms <= 0:
        return np.nan

    return 20 * np.log10(signal_peak / noise_rms)


def compute_frequency_domain_snr(
        time: np.ndarray,
        trace: np.ndarray,
        noise_band: tuple[float, float] = (5.0, 10.0),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate frequency-dependent spectral SNR using a high-frequency
    noise-floor region.

    Assumes time is expressed in seconds.
    """
    time = np.asarray(time)
    trace = np.asarray(trace)

    dt = np.mean(np.diff(time))

    # Remove DC offset
    trace = trace - np.mean(trace)

    # Fourier transform
    spectrum = np.fft.rfft(trace)
    frequencies = np.fft.rfftfreq(len(trace), d=dt)

    amplitude = np.abs(spectrum)

    # Estimate noise from 5–10 THz
    noise_mask = (
            (frequencies >= noise_band[0])
            & (frequencies <= noise_band[1])
    )

    if not np.any(noise_mask):
        raise ValueError(
            "Noise band is outside the available frequency range."
        )

    noise_rms = np.sqrt(
        np.mean(amplitude[noise_mask] ** 2)
    )

    if noise_rms <= 0:
        raise ValueError("Estimated noise RMS must be positive.")

    snr_db = 20 * np.log10(
        np.maximum(amplitude, np.finfo(float).tiny) / noise_rms
    )

    return frequencies, snr_db


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
    """Extract the refractive-index spectrum of a uniform sample slab.

    The sample and reference receive a common time-domain window before
    ``thzpy.transferfunctions.uniform_slab`` is applied. Thickness ``d_mm`` is
    in millimetres and frequency limits are in terahertz.

    Returns:
        ``(frequency_thz, real_refractive_index, complex_refractive_index)``.
    """
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


def compute_density_uncertainty_components(
        mass_kg: float,
        mass_err_kg: float,
        thickness_mm: float,
        thickness_err_mm: float,
        settings: AnalysisSettings,
) -> Tuple[float, float, float]:
    """Return density and its independent and shared-radius standard uncertainties."""
    thickness_m = thickness_mm / 1000.0
    thickness_err_m = thickness_err_mm / 1000.0
    radius_m = settings.radius_m
    radius_err_m = settings.radius_err_m
    if mass_kg <= 0 or thickness_m <= 0 or radius_m <= 0:
        raise ValueError("Mass, thickness, and radius must be positive.")
    if mass_err_kg < 0 or thickness_err_m < 0 or radius_err_m < 0:
        raise ValueError("Standard uncertainties must be non-negative.")

    volume_m3 = np.pi * radius_m ** 2 * thickness_m
    density_kg_m3 = mass_kg / volume_m3
    mass_and_thickness_err_kg_m3 = np.sqrt(
        (mass_err_kg / volume_m3) ** 2
        + (density_kg_m3 * thickness_err_m / thickness_m) ** 2
    )
    radius_err_kg_m3 = 2 * density_kg_m3 * radius_err_m / radius_m
    return (
        density_kg_m3 / 1000.0,
        mass_and_thickness_err_kg_m3 / 1000.0,
        radius_err_kg_m3 / 1000.0,
    )


def compute_density(
        mass_kg: float,
        thickness_mm: float,
        settings: AnalysisSettings,
        *,
        mass_err_kg: float | None = None,
        thickness_err_mm: float | None = None,
) -> Tuple[float, float]:
    """Calculate density and standard uncertainty for independent input quantities."""
    density, independent_err, radius_err = compute_density_uncertainty_components(
        mass_kg=mass_kg,
        mass_err_kg=settings.mass_err_kg if mass_err_kg is None else mass_err_kg,
        thickness_mm=thickness_mm,
        thickness_err_mm=settings.thickness_err_mm if thickness_err_mm is None else thickness_err_mm,
        settings=settings,
    )
    return density, float(np.hypot(independent_err, radius_err))


def compute_refractive_index_error(refractive_index_at_1thz: float, thickness_mm: float,
                                   thickness_err_mm: float, n_medium: float = 1.0) -> float:
    """Thickness contribution to n uncertainty for n = n_medium + C / thickness."""
    if thickness_mm <= 0:
        raise ValueError("Thickness must be positive.")
    if thickness_err_mm < 0:
        raise ValueError("Thickness uncertainty must be non-negative.")
    return abs(refractive_index_at_1thz - n_medium) / thickness_mm * thickness_err_mm


@lru_cache(maxsize=1)
def tao_ice_refractive_index_at_1thz() -> float:
    """Linearly interpolate the Tao et al. solid-ice table at exactly 1 THz."""
    with TAO_ICE_REFRACTIVE_INDEX_PATH.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    frequencies = np.asarray([float(row["f"]) for row in rows], dtype=float)
    indices = np.asarray([float(row["n"]) for row in rows], dtype=float)
    if frequencies.size < 2 or np.any(np.diff(frequencies) <= 0):
        raise ValueError(f"Invalid Tao et al. refractive-index table: {TAO_ICE_REFRACTIVE_INDEX_PATH}")
    if not frequencies[0] <= 1.0 <= frequencies[-1]:
        raise ValueError("The Tao et al. table does not bracket 1 THz.")
    return float(np.interp(1.0, frequencies, indices))


def bruggeman_eps(volume_fraction_ice: np.ndarray, eps_host: complex, eps_ice: complex) -> np.ndarray:
    """Evaluate the physical branch of the symmetric Bruggeman solution."""
    b_term = 3 * volume_fraction_ice * (eps_ice - eps_host) + 2 * eps_host - eps_ice
    return (b_term + np.sqrt(b_term ** 2 + 8 * eps_host * eps_ice)) / 4


def theoretical_emt_curve(
        solid_ice_density_g_cm3: float,
        model: str,
        point_count: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct a density-versus-index EMT curve at exactly 1 THz.

    The ice volume fraction is identified with bulk density divided by the
    supplied solid-ice density. Solid-ice permittivity uses the Tao et al.
    refractive index interpolated at 1 THz.
    """
    density_axis = np.linspace(0.0, solid_ice_density_g_cm3, point_count)
    volume_fraction_ice = np.clip(density_axis / solid_ice_density_g_cm3, 0.0, 1.0)
    eps_ice = tao_ice_refractive_index_at_1thz() ** 2

    if model != "bruggeman":
        raise ValueError(f"Unsupported EMT model: {model}")
    eps_eff = bruggeman_eps(volume_fraction_ice, EPS_VACUUM, eps_ice)

    return density_axis, np.real(np.sqrt(eps_eff))


def emt_model_label(model: str) -> str:
    """Return the publication-facing label for an internal EMT model key."""
    if model == "bruggeman":
        return "Bruggeman"
    raise ValueError(f"Unsupported EMT model: {model}")


def analyze_campaign(
        campaign: CampaignConfig,
        overview_ax: Sequence[plt.Axes],
        scatter_ax: plt.Axes,
        trace_cache: Dict[Path, Tuple[np.ndarray, np.ndarray, int]],
) -> List[MeasurementResult]:
    """Analyze all non-ignored measurements and add them to campaign plots."""
    settings = campaign.analysis
    scatter_color = campaign.plot_color
    fit_band_min, fit_band_max = settings.fit_band_thz

    results: List[MeasurementResult] = []

    SNR_sample = []
    SNR_ref = []
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
            mass_err_kg=measurement.mass_err_kg,
            thickness_err_mm=measurement.thickness_err_mm,
        )

        # Estimate sample spectral SNR
        snr_freqs, sample_snr_db = compute_frequency_domain_snr(
            sample_t,
            sample_p,
            noise_band=(5.0, 10.0),
        )

        # Estimate reference spectral SNR
        ref_snr_freqs, reference_snr_db = compute_frequency_domain_snr(
            reference_t,
            reference_p,
            noise_band=(5.0, 10.0),
        )
        # Calculate maximum SNR for sample
        sample_max_idx = np.nanargmax(sample_snr_db)
        sample_snr_max = sample_snr_db[sample_max_idx]
        sample_snr_max_freq = snr_freqs[sample_max_idx]

        # Calculate maximum SNR for reference
        reference_max_idx = np.nanargmax(reference_snr_db)
        reference_snr_max = reference_snr_db[reference_max_idx]
        reference_snr_max_freq = ref_snr_freqs[reference_max_idx]

        print(
            f"{campaign.campaign_id} | {measurement.measurement_id}\n"
            f"  Sample max SNR:    {sample_snr_max:.1f} dB "
            f"at {sample_snr_max_freq:.2f} THz\n"
            f"  Reference max SNR: {reference_snr_max:.1f} dB "
            f"at {reference_snr_max_freq:.2f} THz"
        )

        SNR_sample.append(sample_snr_max)
        SNR_ref.append(reference_snr_max)

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

    return results, SNR_sample, SNR_ref


def plot_shared_emt_curves(scatter_ax: plt.Axes, results: Sequence[MeasurementResult]) -> Tuple[float, float, int]:
    """Add the Bruggeman curve using the fixed solid-ice density.

    Returns:
        ``(tao_index_at_1thz, solid_density_g_cm3, measured_solid_count)``.
    """
    solid_results = [
        result for result in results
        if result.label.upper() == "SOLID"
    ]
    tao_index = tao_ice_refractive_index_at_1thz()

    model_styles = {
        "bruggeman": {"color": "black", "linestyle": "--"},
    }
    for model in EMT_MODELS:
        emt_density_axis, emt_refractive_index = theoretical_emt_curve(
            solid_ice_density_g_cm3=SOLID_ICE_DENSITY_G_CM3,
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

    return tao_index, SOLID_ICE_DENSITY_G_CM3, len(solid_results)


def write_summary_csv(results: Iterable[MeasurementResult], output_path: Path) -> None:
    """Write the campaign-level density and refractive-index summary table."""
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
    """Resolve explicit config arguments or discover all JSON configs."""
    if config_files:
        return [Path(item).resolve() for item in config_files]
    return sorted(config_dir.glob("*.json"))


def make_unique_legend(ax: plt.Axes) -> None:
    """Create an axes legend with only the first handle for each label."""
    handles, labels = ax.get_legend_handles_labels()
    unique: Dict[str, object] = {}
    for handle, label in zip(handles, labels):
        if label not in unique:
            unique[label] = handle
    ax.legend(unique.values(), unique.keys())


def parse_args() -> argparse.Namespace:
    """Parse command-line options for campaign analysis."""
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
    """Run all selected campaigns and save overview plots and a CSV table."""
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

    SNR_sample = []
    SNR_ref = []


    for campaign in campaigns:
        overview_fig, overview_axes = plt.subplots(nrows=2, figsize=(10, 8))
        result, snr_sample, snr_reference = analyze_campaign(campaign, overview_axes, scatter_ax, trace_cache)
        SNR_sample.extend(snr_sample)
        SNR_ref.extend(snr_reference)

        all_results.extend(result)

        overview_axes[0].set_title(campaign.campaign_id)
        overview_axes[0].set_xlabel("Time (ps)")
        overview_axes[0].set_ylabel("Amplitude (a.u.)")
        overview_axes[1].set_xlabel("Frequency (THz)")
        overview_axes[1].set_ylabel("Refractive index (-)")
        make_unique_legend(overview_axes[0])
        overview_fig.tight_layout()
        save_figure_outputs(
            overview_fig,
            args.output_dir / f"{campaign.campaign_id}_trace_overview",
        )
        overview_figures.append(overview_fig)

        if campaign.campaign_id == "dataset 5":
            detail_fig, (detail_ax, absorption_ax) = plt.subplots(
                2, 1,
                figsize=(16, 11),
                sharex=True,
            )

            measurement_lookup = {
                m.measurement_id: m
                for m in campaign.measurements
                if not m.ignore
            }

            result_lookup = {
                r.measurement_id: r
                for r in all_results
                if r.campaign_id == campaign.campaign_id
            }

            lower_f = 0.5
            noise_floor = 1.6

            sorted_measurements = []

            for line in overview_axes[1].get_lines():
                measurement_id = line.get_label()

                if (
                        measurement_id not in measurement_lookup
                        or measurement_id not in result_lookup
                ):
                    continue

                measurement = measurement_lookup[measurement_id]
                result = result_lookup[measurement_id]

                sorted_measurements.append(
                    (result.density_g_cm3, line, measurement)
                )

            # Sort from highest to lowest density
            sorted_measurements.sort(
                key=lambda item: item[0],
                reverse=True,
            )

            c = 299792458.0  # Speed of light [m/s]

            for density, line, measurement in sorted_measurements:
                # Load sample and reference traces
                sample_t, sample_p, _ = read_trace_cached(
                    measurement.path,
                    trace_cache,
                )

                reference_t, reference_p, _ = read_trace_cached(
                    measurement.reference_path,
                    trace_cache,
                )

                # Extract complex refractive index
                f, n, complex_n = get_refraction_index(
                    sample_t,
                    sample_p,
                    reference_t,
                    reference_p,
                    window_half_width=campaign.analysis.window_half_width,
                    win_func=campaign.analysis.window_function,
                    min_frequency=campaign.analysis.min_frequency_thz,
                    max_frequency=campaign.analysis.max_frequency_thz,
                    d_mm=measurement.thickness_mm,
                )

                # Extinction coefficient
                kappa = np.imag(complex_n)

                # Absorption coefficient [cm^-1]
                alpha = (
                        4 * np.pi * f * 1e12 * kappa
                        / (c * 100)
                )

                valid_mask = (f >= lower_f) & (f <= noise_floor)
                low_mask = f < lower_f
                high_mask = f > noise_floor

                # Refractive index
                plotted_line, = detail_ax.plot(
                    f[valid_mask],
                    n[valid_mask],
                    linewidth=2,
                    label=(
                        f"d = {measurement.thickness_mm:.2f} mm\n"
                        f"ρ = {density:.3f} g/cm³"
                    ),
                )

                color = plotted_line.get_color()

                detail_ax.plot(
                    f[low_mask], n[low_mask],
                    linestyle="--",
                    linewidth=1.2,
                    color=color,
                )

                detail_ax.plot(
                    f[high_mask], n[high_mask],
                    linestyle="--",
                    linewidth=1.2,
                    color=color,
                )

                # Absorption coefficient
                absorption_ax.plot(
                    f[valid_mask],
                    alpha[valid_mask],
                    linewidth=2,
                    color=color,
                )

                absorption_ax.plot(
                    f[low_mask], alpha[low_mask],
                    linestyle="--",
                    linewidth=1.2,
                    color=color,
                )

                absorption_ax.plot(
                    f[high_mask], alpha[high_mask],
                    linestyle="--",
                    linewidth=1.2,
                    color=color,
                )

            fit_min, fit_max = campaign.analysis.fit_band_thz

            for ax in (detail_ax, absorption_ax):
                ax.axvspan(
                    fit_min,
                    fit_max,
                    color="blue",
                    alpha=0.12,
                    zorder=0,
                )
                ax.grid(True, alpha=0.3)

            detail_ax.set_ylabel(r"Refractive index $n$ [-]")
            absorption_ax.set_ylabel(
                r"Absorption coefficient $\alpha$ [cm$^{-1}$]"
            )
            absorption_ax.set_xlabel(r"$f$ [THz]")

            # Legend outside
            detail_ax.legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
            )

            detail_fig.tight_layout(rect=(0, 0, 0.95, 1))

            save_figure_outputs(
                detail_fig,
                args.output_dir / "subset_detailed_refractive_index_absorption",
            )

            overview_figures.append(detail_fig)

    snr_sample_avg = np.mean(SNR_sample)
    snr_sample_std = np.std(SNR_sample)
    snr_ref_avg = np.mean(SNR_ref)
    snr_ref_std = np.std(SNR_ref)

    print(
        f"  Sample SNR:    {snr_sample_avg:.1f} +/- {snr_sample_std:.1f} dB\n"
        f"  Reference SNR:    {snr_ref_avg:.1f} +/- {snr_ref_std:.1f} dB\n"
    )


    _, solid_density_g_cm3, solid_result_count = plot_shared_emt_curves(scatter_ax, all_results)
    scatter_ax.set_xlabel(r"Ice density [g/cm$^3$]")
    scatter_ax.set_ylabel("Refractive index @ 1 THz (-)")
    scatter_ax.set_title(
        f"Measured points with Bruggeman EMT "
        f"({solid_result_count} measured solids, fixed density {solid_density_g_cm3:.3f} g/cm^3)"
    )
    make_unique_legend(scatter_ax)

    write_summary_csv(all_results, args.output_dir / "campaign_analysis_summary.csv")

    scatter_fig.tight_layout()
    save_figure_outputs(
        scatter_fig,
        args.output_dir / "campaign_refractive_index_vs_density",
    )

    if args.show:
        plt.show()
    else:
        for overview_fig in overview_figures:
            plt.close(overview_fig)
        plt.close(scatter_fig)


if __name__ == "__main__":
    main()
