from pathlib import Path

import numpy as np
from math_utils import get_fft
from pydotthz import DotthzFile
from thzpy.timedomain import common_window
from thzpy.transferfunctions import uniform_slab
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from matplotlib import colors

from utils import get_thz_files

eps_air = 1.0  # vacuum
solid_ice_density = 0.918  # g/cm^3


def maxwell_garnett_f(eps_eff, eps_h, eps_i):
    num = (eps_eff - eps_h) * (eps_i + 2 * eps_h)
    den = (eps_i - eps_h) * (eps_eff + 2 * eps_h)
    return num / den


def looyenga_f(eps_eff, eps_h, eps_i):
    return (np.power(eps_eff, 1 / 3) - np.power(eps_h, 1 / 3)) / (
        np.power(eps_i, 1 / 3) - np.power(eps_h, 1 / 3)
    )


def physical_fraction(v):
    v_real = np.real(v)
    return np.clip(v_real, 0.0, 1.0)


def bruggeman_vi(epsilon_eff, epsilon_i):
    epsilon_h = 1.0

    LHS = (epsilon_h - epsilon_eff) / (epsilon_h + 2 * epsilon_eff)
    RHS = (epsilon_i - epsilon_eff) / (epsilon_i + 2 * epsilon_eff)
    v_i = LHS / (LHS - RHS)
    return v_i


def emt_density_correct(complex_n_eff, freq_thz, density_g_cm3):
    """
    Invert Looyenga EMT to estimate intrinsic (solid-ice-equivalent) optical constants
    from porous effective-medium data.
    """
    ice_fraction = np.clip(density_g_cm3 / solid_ice_density, 1e-6, 1.0)
    eps_eff = complex_n_eff ** 2
    eps_air_cuberoot = np.power(eps_air, 1 / 3)

    eps_ice_cuberoot = (np.power(eps_eff, 1 / 3) - (1.0 - ice_fraction) * eps_air_cuberoot) / ice_fraction
    eps_ice = np.power(eps_ice_cuberoot, 3)
    n_ice = np.sqrt(eps_ice)

    c0 = 299792458.0  # m/s
    alpha_cm_inv = 4.0 * np.pi * (freq_thz * 1e12) * np.maximum(n_ice.imag, 0.0) / (c0 * 100.0)

    return n_ice.real, alpha_cm_inv


def get_thz_data(path: Path):
    thz_files = get_thz_files(path)
    for file in thz_files:
        with DotthzFile(file) as f:
            t = f["Single Pixel 0"].datasets["Sample"][:, 0]
            d = f["Single Pixel 0"].datasets["Sample"][:, 1]

            d = d[(t < 1960) & (t > 1860)]
            t = t[(t < 1960) & (t > 1860)]
            return t, d


def load_img(path: Path):
    thz_files = get_thz_files(path)
    for file in thz_files:
        with DotthzFile(file) as f:
            t = f["Image"].datasets["time"][:]
            d = f["Image"].datasets["dataset"][:]

            d = d[:, :, t < 1960]
            t = t[t < 1960]
            return t, d


def get_refraction_index(
    time,
    traces,
    t_ref,
    p_ref,
    window_half_width=15,
    win_func="hanning",
    min_frequency=0.2,
    max_frequency=3,
    d_mm=1.0,
    mask_radius=None,
):
    data_ref = np.array([t_ref, p_ref])

    freq = None
    freq_len = None

    p_pair = np.array([time, traces])
    try:
        sample, reference = common_window(
            [p_pair, data_ref], half_width=window_half_width, win_func=win_func
        )

        buffer_pixel = uniform_slab(
            d_mm,
            sample,
            reference,
            n_med=1,
            upsampling=1,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            all_optical_constants=True,
        )

        freq = np.array(buffer_pixel["frequency"])
        freq_len = len(freq)

    except Exception:
        pass

    if freq is None:
        raise RuntimeError("Unable to compute frequency axis.")

    refractive_index = np.full(freq_len, np.nan)
    absorption_coefficient = np.full(freq_len, np.nan)
    complex_refractive_index = np.full(freq_len, np.nan + 0j)

    try:
        sample, reference = common_window(
            [p_pair, data_ref], half_width=window_half_width, win_func=win_func
        )

        buffer_pixel = uniform_slab(
            d_mm,
            sample,
            reference,
            n_med=1,
            upsampling=1,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            all_optical_constants=True,
        )

        freq_pix = np.array(buffer_pixel["frequency"])
        n = np.array(buffer_pixel["refractive_index"]).astype(complex)
        alpha = np.array(buffer_pixel["absorption_coefficient"]).astype(float)

        if len(freq_pix) == freq_len:
            complex_refractive_index = n
            refractive_index = n.real
            absorption_coefficient = alpha

    except Exception:
        pass

    return freq, refractive_index, complex_refractive_index, absorption_coefficient


if __name__ == "__main__":

    fig, axes = plt.subplots(
        nrows=3,
        figsize=(11, 9),
        sharex=True,
        constrained_layout=True,
    )

    solid_ice_path = Path(
        "/Users/linus/Documents/collimated_solid_ice_3a/data/single_pixel"
    )

    t_solid, p_solid = get_thz_data(solid_ice_path)

    silicon_ref_path = Path(
        "/Users/linus/Documents/collimated_silicon_metal_sheet_fix_focus/data/single_pixel"
    )
    t_ref, p_ref = get_thz_data(silicon_ref_path)

    measurements = [
        (Path("/Users/linus/Documents/collimated_solid_ice_2/data/single_pixel"), "SOLID", 918 * 2.8274333882308137e-05,
         10),
        (Path("/Users/linus/Documents/collimated_solid_ice_3/data/single_pixel"), "SOLID", 918 * 2.8274333882308137e-05,
         10),
        (Path("/Users/linus/Documents/collimated_solid_ice_3a/data/single_pixel"), "SOLID",
         918 * 2.8274333882308137e-05, 10),
        (Path("/Users/linus/Documents/collimated_spipa_b/data/single_pixel"), "SPIPA-B", 0.01550, 10),
        (Path("/Users/linus/Documents/collimated_spipa_b_2/data/single_pixel"), "SPIPA-B", 0.01550, 10),
        (Path("/Users/linus/Documents/collimated_spipa_b_3/data/single_pixel"), "SPIPA-B", 0.01550, 9.5),
        (Path("/Users/linus/Documents/collimated_spipa_b_4/data/single_pixel"), "SPIPA-B", 0.01550, 9.5),
        (Path("/Users/linus/Documents/collimated_spipa_b_5/data/single_pixel"), "SPIPA-B", 0.014811, 10),
        (Path("/Users/linus/Documents/collimated_spipa_b_6/data/single_pixel"), "SPIPA-B", 0.014811, 10),
        # (Path("/Users/linus/Documents/collimated_spipa_b_7/data/single_pixel"), "SPIPA-B", 0.01760, 10),
        # (Path("/Users/linus/Documents/collimated_spipa_b_8/data/single_pixel"), "SPIPA-B", 0.01760, 10),
        (Path("/Users/linus/Documents/collimated_spipa_b_9/data/single_pixel"), "SPIPA-B", 0.01760, 10),
        (Path("/Users/linus/Documents/collimated_spipa_b_10/data/single_pixel"), "SPIPA-B", 0.01760, 10),
        (Path("/Users/linus/Documents/collimated_frost/data/single_pixel"), "Frost", 0.00320, 10),
        (Path("/Users/linus/Documents/collimated_frost_2/data/single_pixel"), "Frost", 0.00320, 10),
        (Path("/Users/linus/Documents/collimated_frost_3/data/single_pixel"), "Frost", 0.00482, 10),
        (Path("/Users/linus/Documents/collimated_frost_4/data/single_pixel"), "Frost", 0.00482, 10),
        (Path("/Users/linus/Documents/collimated_frost_5/data/single_pixel"), "Frost", 0.00831, 10.5),
        (Path("/Users/linus/Documents/collimated_frost_6/data/single_pixel"), "Frost", 0.00831, 10.5),
    ]

    freqs_solid, refractive_index_solid, complex_refractive_index_solid, absorption_solid = get_refraction_index(
        t_solid,
        p_solid,
        t_ref,
        p_ref,
        window_half_width=25,
        win_func="hanning",
        min_frequency=0.2,
        max_frequency=3,
        d_mm=10.0,
    )

    refractive_indices = []
    densities = []
    refractive_indices_err = []
    densities_err = []

    # Build a density-based blue->red color scale for all plotted traces.
    density_values = []
    absorption_curves = []
    solid_absorption_curves = []

    for measurement in measurements:
        r = 0.03
        h = measurement[3] / 1000
        volume = np.pi * r ** 2 * h
        mass = measurement[2]
        density_values.append((mass / volume) / 1e3)

    density_norm = colors.Normalize(
        vmin=min(density_values),
        vmax=max(density_values),
    )
    density_cmap = colors.LinearSegmentedColormap.from_list(
        "density_blue_red", ["blue", "red"]
    )

    for measurement in measurements:

        r_err = 0.0001
        h_err = 0.0005
        m_err = 0.00001
        r = 0.03

        h = measurement[3] / 1000
        volume = np.pi * r ** 2 * h

        mass = measurement[2]
        density = mass / volume
        density_g_cm3 = density / 1e3
        line_color = density_cmap(density_norm(density_g_cm3))

        density_err = np.sqrt(
            (m_err / volume) ** 2
            + (mass / (np.pi * r ** 2 * h ** 2) * h_err) ** 2
            + (2 * mass / (np.pi * r ** 3 * h) * r_err) ** 2
        )

        t, p = get_thz_data(measurement[0])

        freqs, refractive_index, complex_refractive_index, _ = get_refraction_index(
            t,
            p,
            t_ref,
            p_ref,
            window_half_width=30,
            win_func="hanning",
            min_frequency=0.2,
            max_frequency=3,
            d_mm=measurement[3],
        )

        refractive_index_corr, absorption_corr = emt_density_correct(
            complex_refractive_index, freqs, density_g_cm3
        )

        axes[0].plot(
            freqs, refractive_index_corr, label=f"EMT n ρ={density_g_cm3:.3f}", color=line_color
        )

        axes[1].plot(
            freqs,
            absorption_corr,
            label=f"EMT α ρ={density_g_cm3:.3f}",
            color=line_color,
        )

        absorption_curves.append(
            {
                "freqs": freqs,
                "absorption": absorption_corr,
                "label": f"α/α_solid ρ={density_g_cm3:.3f}",
                "color": line_color,
                "is_solid": measurement[1] == "SOLID",
            }
        )
        if measurement[1] == "SOLID":
            solid_absorption_curves.append(absorption_corr)

        n = np.mean(refractive_index_corr[(freqs > 0.9) & (freqs < 1.1)])

        refractive_indices.append(n)
        densities.append(density_g_cm3)

        refractive_indices_err.append(n / h * h_err)
        densities_err.append(density_err / 1000)

    if len(solid_absorption_curves) > 0:
        solid_absorption_avg = np.nanmean(np.array(solid_absorption_curves), axis=0)
    else:
        solid_absorption_avg = np.full_like(absorption_curves[0]["absorption"], np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        for curve in absorption_curves:
            normalized_absorption = curve["absorption"] / solid_absorption_avg
            axes[2].plot(
                curve["freqs"],
                normalized_absorption,
                label=curve["label"],
                color=curve["color"],
            )

    axes[0].set_ylabel("Refractive Index [-]")
    axes[1].set_ylabel("Absorption Coefficient")
    axes[2].set_xlabel("Frequency (THz)")
    axes[2].set_ylabel(r"$\alpha / \langle\alpha_{solid}\rangle$ [-]")

    axes[2].set_ylim(0, 40)

    # axes[0].legend(fontsize=8)
    # axes[1].legend(fontsize=8)
    # axes[2].legend(fontsize=8)

    def thz_to_mm(freq_thz):
        freq_thz = np.asarray(freq_thz, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return 0.299792458 / freq_thz

    def mm_to_thz(wavelength_mm):
        wavelength_mm = np.asarray(wavelength_mm, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return 0.299792458 / wavelength_mm

    for ax in axes:
        secax = ax.secondary_xaxis("top", functions=(thz_to_mm, mm_to_thz))
        secax.set_xlabel("Wavelength (mm)")

    density_sm = plt.cm.ScalarMappable(norm=density_norm, cmap=density_cmap)
    density_sm.set_array([])
    density_cbar = fig.colorbar(
        density_sm, ax=axes, location="right", fraction=0.05, pad=0.03
    )
    density_cbar.set_label(r"Density [g/cm$^3$]")

    # Enforce identical x-limits after all artists/secondary axes are added.
    # for ax in axes:
    #     ax.autoscale(enable=False, axis="x")
    #     ax.set_xbound(0.2, 2.5)
    #     ax.set_xlim(0.2, 2.5)
    #     ax.xaxis.set_major_locator(FixedLocator([0.2, 0.5, 1.0, 1.5, 2.0, 2.5]))
    #     ax.margins(x=0)

    plt.show()

    # plt.errorbar(
    #     densities,
    #     refractive_indices,
    #     xerr=densities_err,
    #     yerr=refractive_indices_err,
    #     fmt="o",
    #     color="red",
    #     ecolor="black",
    #     capsize=2,
    #     label="Measured Data",
    # )
    #
    # plt.xlabel(r"Ice Density [g/cm$^3$]")
    # plt.ylabel("Refractive Index @ 1 THz [-]")
    # plt.legend()
    #
    # plt.savefig("refractive_index_vs_density.png", dpi=300)
    # plt.savefig("refractive_index_vs_density.pdf")
    #
    # plt.show()
