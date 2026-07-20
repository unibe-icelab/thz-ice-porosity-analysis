from pathlib import Path

from matplotlib import pyplot as plt
from pydotthz import DotthzFile

from math_utils import get_fft
from utils import get_thz_files

import numpy as np
import miepython

C = 299792458


def _to_array_like(value, template):
    if np.isscalar(value):
        return np.full_like(template, float(value), dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.shape != template.shape:
        raise ValueError("Input array must match frequency shape.")
    return arr


def mie_extinction_coefficient(
        freq,
        diameters,
        weights,
        number_density,
        n_particle=1.5 + 0j,
        n_host=1.0,
):
    """Independent-scattering Mie extinction coefficient mu_mie [1/m]."""

    weights = weights / np.sum(weights)
    Cext = np.zeros_like(freq)

    for d, w in zip(diameters, weights):
        r = d / 2

        for i, f in enumerate(freq):
            if f <= 0:
                continue

            wavelength = C / (f * 1e12)
            m = n_particle / n_host

            # convert meters -> microns for miepython.efficiencies
            d_um = d * 1e6
            lambda_um = wavelength * 1e6

            qext, qsca, qback, g = miepython.efficiencies(m, d_um, lambda_um)
            Cext[i] += w * qext * np.pi * r ** 2

    return number_density * Cext


def apply_structured_scattering_transfer(
        freq,
        spectrum,
        diameters,
        weights,
        number_density,
        thickness=10e-3,
        n_particle=1.5 + 0j,
        n_host=1.0,
        alpha_absorption=0.0,
        porosity=0.0,
        dep_strength=0.0,
        dep_exp=2.0,
        empirical_strength=0.0,
        empirical_exp=3.0,
        path_base=0.0,
        path_freq_strength=0.0,
        path_freq_exp=2.0,
        freq_ref_thz=1.0,
):
    """
    Extinction model:
    - EMT/bulk absorption: alpha_absorption [1/m]
    - Mie independent scattering: mu_mie [1/m]
    - Dependent-scattering amplification: multiplicative boost on mu_mie
    - Extra microstructure term: empirical A*(f/f_ref)^b scaled by porosity
    - Multiple-scattering path enhancement: L_eff >= thickness
    """

    mu_mie = mie_extinction_coefficient(
        freq=freq,
        diameters=diameters,
        weights=weights,
        number_density=number_density,
        n_particle=n_particle,
        n_host=n_host,
    )

    f_scaled = np.clip(freq / freq_ref_thz, 0.0, None)
    dep_factor = 1.0 + dep_strength * porosity * np.power(f_scaled, dep_exp)
    dep_factor = np.maximum(dep_factor, 0.0)

    mu_dep = mu_mie * dep_factor
    mu_emp = empirical_strength * porosity * np.power(f_scaled, empirical_exp)

    alpha_abs = _to_array_like(alpha_absorption, freq)
    mu_total = alpha_abs + mu_dep + mu_emp

    path_factor = 1.0 + path_base * porosity + path_freq_strength * porosity * np.power(f_scaled, path_freq_exp)
    path_factor = np.maximum(path_factor, 1e-9)
    effective_thickness = thickness * path_factor

    # For field amplitude, use half the intensity attenuation exponent.
    H = np.exp(-mu_total * effective_thickness / 2.0)
    spectrum_out = spectrum * H

    details = {
        "mu_mie": mu_mie,
        "mu_dep": mu_dep,
        "mu_emp": mu_emp,
        "alpha_abs": alpha_abs,
        "mu_total": mu_total,
        "path_factor": path_factor,
        "effective_thickness": effective_thickness,
    }
    return spectrum_out, details


def apply_scattering_transfer(
        freq,
        spectrum,
        diameters,
        weights,
        number_density,
        thickness=10e-3,
        n_particle=1.5 + 0j,
        n_host=1.0,
):
    """Backward-compatible interface using pure independent Mie extinction."""
    spectrum_scattered, details = apply_structured_scattering_transfer(
        freq=freq,
        spectrum=spectrum,
        diameters=diameters,
        weights=weights,
        number_density=number_density,
        thickness=thickness,
        n_particle=n_particle,
        n_host=n_host,
    )
    return spectrum_scattered, details["mu_total"]


def get_thz_data(path: Path):
    thz_files = get_thz_files(path)
    for file in thz_files:
        with DotthzFile(file) as f:
            t = f["Single Pixel 0"].datasets["Sample"][:, 0]
            d = f["Single Pixel 0"].datasets["Sample"][:, 1]

            d = d[(t < 1960) & (t > 1860)]
            t = t[(t < 1960) & (t > 1860)]
            return t, d


if __name__ == "__main__":

    silicon_ref_path = Path(
        "/Users/linus/Documents/collimated_silicon_metal_sheet_fix_focus/data/single_pixel"
    )

    t_ref, p_ref = get_thz_data(silicon_ref_path)

    f, a, arg = get_fft(t_ref, p_ref)

    spectrum = a * np.exp(1j * arg)

    # particle size distribution: mean 0.2 mm, sigma 0.1 mm
    diameters = np.linspace(0.01e-3, 0.60e-3, 300)
    mean_d = 0.2e-3
    sigma_d = 0.1e-3
    weights = np.exp(-(diameters - mean_d) ** 2 / (2 * sigma_d ** 2))
    weights /= np.sum(weights)

    # Keep this at your EMT-corrected true absorption in [1/m] if available.
    alpha_emt = np.zeros_like(f, dtype=float)

    scenarios = [
        {
            "name": "Solid-like (minimal structure loss)",
            "color": "tab:blue",
            "number_density": 5e7,
            "porosity": 0.05,
            "dep_strength": 0.2,
            "dep_exp": 2.0,
            "empirical_strength": 5.0,
            "empirical_exp": 3.0,
            "path_base": 0.1,
            "path_freq_strength": 0.1,
            "path_freq_exp": 2.0,
        },
        {
            "name": "Porous-like (50% density)",
            "color": "tab:orange",
            "number_density": 4e8,
            "porosity": 0.5,
            "dep_strength": 2.0,
            "dep_exp": 2.2,
            "empirical_strength": 40.0,
            "empirical_exp": 3.0,
            "path_base": 0.7,
            "path_freq_strength": 0.8,
            "path_freq_exp": 2.0,
        },
        {
            "name": "Frost-like (10% density, irregular)",
            "color": "tab:red",
            "number_density": 9e8,
            "porosity": 0.9,
            "dep_strength": 4.0,
            "dep_exp": 2.5,
            "empirical_strength": 120.0,
            "empirical_exp": 3.5,
            "path_base": 1.8,
            "path_freq_strength": 2.0,
            "path_freq_exp": 2.3,
        },
    ]

    positive_freq = f > 0
    nonzero_amp = np.abs(a) > 1e-20
    valid = positive_freq & nonzero_amp

    fig, axes = plt.subplots(nrows=3, sharex=False, figsize=(12, 10))

    axes[0].plot(diameters * 1e3, weights, color="tab:green")
    axes[0].set_xlabel("Diameter (mm)")
    axes[0].set_ylabel("Probability")
    axes[0].set_title("Particle size distribution (mean 0.2 mm, sigma 0.1 mm)")

    axes[1].semilogy(f, np.ones_like(f), color="black", linestyle="--", linewidth=1.0, label="No loss baseline")
    axes[1].set_xlabel("Frequency (THz)")
    axes[1].set_ylabel("Amplitude transmission")

    for scenario in scenarios:
        spectrum_model, details = apply_structured_scattering_transfer(
            freq=f,
            spectrum=spectrum,
            diameters=diameters,
            weights=weights,
            number_density=scenario["number_density"],
            thickness=10e-3,
            n_particle=1.5,
            n_host=1.0,
            alpha_absorption=alpha_emt,
            porosity=scenario["porosity"],
            dep_strength=scenario["dep_strength"],
            dep_exp=scenario["dep_exp"],
            empirical_strength=scenario["empirical_strength"],
            empirical_exp=scenario["empirical_exp"],
            path_base=scenario["path_base"],
            path_freq_strength=scenario["path_freq_strength"],
            path_freq_exp=scenario["path_freq_exp"],
            freq_ref_thz=1.0,
        )

        model_amp = np.abs(spectrum_model)
        ratio = np.full_like(model_amp, np.nan, dtype=float)
        ratio[valid] = np.abs(model_amp[valid] / a[valid])

        mu_total = details["mu_total"]
        tau_amp = mu_total[valid] * details["effective_thickness"][valid] / 2.0

        print(
            f"{scenario['name']} -> "
            f"mu_total min/max [1/m]: {float(np.nanmin(mu_total[valid])):.4f}, {float(np.nanmax(mu_total[valid])):.4f}"
        )
        print(
            f"{scenario['name']} -> "
            f"amp transmission min/max: {float(np.nanmin(ratio[valid])):.4f}, {float(np.nanmax(ratio[valid])):.4f}"
        )
        print(
            f"{scenario['name']} -> "
            f"tau_amp min/max: {float(np.nanmin(tau_amp)):.4f}, {float(np.nanmax(tau_amp)):.4f}"
        )

        axes[1].semilogy(f, ratio, color=scenario["color"], label=scenario["name"])
        axes[2].plot(f, mu_total, color=scenario["color"], label=scenario["name"])

    axes[2].set_xlabel("Frequency (THz)")
    axes[2].set_ylabel("Total extinction coeff. [1/m]")
    axes[2].set_title("EMT absorption + structure-dependent scattering/extinction")
    axes[1].legend()
    axes[2].legend()

    fig.tight_layout()
    plt.show()
