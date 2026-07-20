from pathlib import Path

import numpy as np
from math_utils import get_fft
from pydotthz import DotthzFile
from thzpy.timedomain import common_window
from thzpy.transferfunctions import uniform_slab
import matplotlib.pyplot as plt

from utils import get_thz_files

eps_air = 1.0  # vacuum


def maxwell_garnett_f(eps_eff, eps_h, eps_i):
    """
    Inverse MG: solve for inclusion fraction f given eps_eff, host eps_h and inclusion eps_i.
    """
    num = (eps_eff - eps_h) * (eps_i + 2 * eps_h)
    den = (eps_i - eps_h) * (eps_eff + 2 * eps_h)
    return num / den


def looyenga_f(eps_eff, eps_h, eps_i):
    """
    Inverse Looyenga: returns fraction of component 'i' when mixture is between h and i.
    eps_h = host (air or ice), eps_i = other component.
    """
    return (np.power(eps_eff, 1 / 3) - np.power(eps_h, 1 / 3)) / (np.power(eps_i, 1 / 3) - np.power(eps_h, 1 / 3))


def physical_fraction(v):
    v_real = np.real(v)
    return np.clip(v_real, 0.0, 1.0)


def bruggeman_vi(epsilon_eff, epsilon_i):
    epsilon_h = 1.0  # vacuum

    # (1 - v_i) * LHS = -v_i * RHS
    # LHS - v_i * LHS = -v_i * RHS
    # LHS = v_i * (LHS - RHS)
    LHS = (epsilon_h - epsilon_eff) / (epsilon_h + 2 * epsilon_eff)
    RHS = (epsilon_i - epsilon_eff) / (epsilon_i + 2 * epsilon_eff)
    v_i = LHS / (LHS - RHS)
    return v_i


def get_thz_data(path: Path):
    thz_files = get_thz_files(path)
    for file in thz_files:
        # there should be only one...
        with DotthzFile(file) as f:
            t = f["Single Pixel 0"].datasets["Sample"][:, 0]
            d = f["Single Pixel 0"].datasets["Sample"][:, 1]

            d = d[(t < 1960) & (t > 1860)]
            t = t[(t < 1960) & (t > 1860)]
            return t, d


def load_img(path: Path):
    thz_filees = get_thz_files(path)
    for file in thz_filees:
        # there should be only one...
        with DotthzFile(file) as f:
            t = f["Image"].datasets["time"][:]
            d = f["Image"].datasets["dataset"][:]

            d = d[:, :, t < 1960]
            t = t[t < 1960]
            return t, d


def get_refraction_index(time, traces, t_ref, p_ref, window_half_width=15, win_func="hanning",
                         min_frequency=0.2, max_frequency=3, d_mm=1.0, mask_radius=None):
    """
    Compute refractive index map for all pixels in `traces`.
    This function first determines the frequency axis from the first valid pixel,
    then preallocates arrays with a homogeneous shape (nx, ny, n_freq) and fills
    them. Failed pixels are filled with NaN to avoid inhomogeneous nested lists.
    """
    data_ref = np.array([t_ref, p_ref])

    # Determine frequency axis from the first pixel that yields a valid result
    freq = None
    freq_len = None

    p_pair = np.array([time, traces])
    try:
        sample, reference = common_window([p_pair, data_ref],
                                          half_width=window_half_width, win_func=win_func)
        buffer_pixel = uniform_slab(d_mm,
                                    sample, reference, n_med=1,
                                    upsampling=1, min_frequency=min_frequency, max_frequency=max_frequency,
                                    all_optical_constants=True)
        freq = np.array(buffer_pixel["frequency"])
        freq_len = len(freq)
    except Exception:
        # try next pixel
        pass

    if freq is None:
        raise RuntimeError("Unable to compute frequency axis from any pixel. Check input traces/ref.")

    # Preallocate arrays with homogeneous shape and fill with NaN for failures
    refractive_index = np.full(freq_len, np.nan, dtype=float)
    absorption_coefficient = np.full(freq_len, np.nan, dtype=float)
    complex_refractive_index = np.full(freq_len, np.nan + 0j, dtype=complex)

    try:

        sample, reference = common_window([p_pair, data_ref],
                                          half_width=window_half_width, win_func=win_func)
        buffer_pixel = uniform_slab(d_mm,
                                    sample, reference, n_med=1,
                                    upsampling=1, min_frequency=min_frequency, max_frequency=max_frequency,
                                    all_optical_constants=True)
        freq_pix = np.array(buffer_pixel["frequency"])
        n = np.array(buffer_pixel["refractive_index"]).astype(complex)
        alpha = np.array(buffer_pixel["absorption_coefficient"]).astype(float)
        # Only accept if frequency axis matches expected length; otherwise fill NaNs
        if len(freq_pix) == freq_len:
            complex_refractive_index = n
            refractive_index = n.real
            absorption_coefficient = alpha
        else:
            # leave as NaN if mismatch
            pass
    except Exception:
        # leave as NaN on any error
        pass

    return freq, refractive_index, complex_refractive_index


if __name__ == '__main__':

    fig, axes = plt.subplots(nrows=2, figsize=(8, 6))

    solid_ice_path = Path("/Users/linus/Documents/collimated_solid_ice_3a/data/single_pixel")

    t_solid, p_solid = get_thz_data(solid_ice_path)

    # p = np.mean(p_solid, axis=(0, 1))
    # f, a, arg = get_fft( t_solid, p)
    #
    # axes[0].semilogy(f, a, label=f"reference solid ice")
    # axes[1].plot(f, arg, label=f"reference solid ice")

    silicon_ref_path = Path("/Users/linus/Documents/collimated_silicon_metal_sheet_fix_focus/data/single_pixel")
    t_ref, p_ref = get_thz_data(silicon_ref_path)

    # f, a, arg = get_fft(t_ref, p_ref)
    #
    # axes[0].semilogy(f, a, label=f"silicon reference")
    # axes[1].plot(f, arg, label=f"silicon reference")

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

    freqs_solid, refractive_index_solid, complex_refractive_index_solid = get_refraction_index(t_solid, p_solid, t_ref,
                                                                                               p_ref,
                                                                                               window_half_width=25,
                                                                                               win_func="hanning",
                                                                                               min_frequency=0.2,
                                                                                               max_frequency=3,
                                                                                               d_mm=10.0,
                                                                                               mask_radius=None)
    n_ice = complex_refractive_index_solid
    epsilon_ice = n_ice ** 2
    solid_ice_density = 0.918  # g/cm3

    # axes[0].plot(freqs_solid, np.mean((complex_refractive_index_solid**2).imag, axis=(0,1)), label=f"solid ice")
    # axes[1].plot(freqs_solid, np.mean((complex_refractive_index_solid**2).imag, axis=(0,1)), label=f"solid ice")

    # axes[0].plot(t_solid, np.mean(p_solid, axis=(0,1)), label="Solid ice")
    # axes[1].plot(freqs_solid, np.nanmean(refractive_index_solid, axis=(0, 1)), label="Solid ice")

    refractive_indices = []
    densities = []

    refractive_indices_err = []
    densities_err = []

    v_i_bgs = []
    v_i_looyengas = []
    v_i_mgs = []


    for measurement in measurements:
        r_err = 0.0001
        h_err = 0.0005
        m_err = 0.00001
        r = 0.03  # m

        h = measurement[3] / 1000  # m
        volume = np.pi * r ** 2 * h  # m^3

        mass = measurement[2]  # kg
        density = mass / volume  # kg/m^3
        density_g_cm3 = density / 1e3  # g/cm^3

        density_err = np.sqrt(
            (m_err / volume) ** 2
            + (mass / (np.pi * r ** 2 * h ** 2) * h_err) ** 2
            + (2 * mass / (np.pi * r ** 3 * h) * r_err) ** 2
        )
        # print(mass, volume, density, density_g_cm3)

        t, p = get_thz_data(measurement[0])

        freqs, refractive_index, complex_refractive_index = get_refraction_index(t, p, t_ref, p_ref,
                                                                                 window_half_width=30,
                                                                                 win_func="hanning",
                                                                                 min_frequency=0.2, max_frequency=3,
                                                                                 d_mm=measurement[3],
                                                                                 mask_radius=None)

        axes[0].plot(t, p, label=f"Porous = {density_g_cm3:.3f}")

        axes[1].plot(freqs, refractive_index, label=f"Porous = {density_g_cm3:.3f}")

        neff = complex_refractive_index
        epsilon_eff = neff ** 2

        n = np.mean(refractive_index[(freqs > 0.9) & (freqs < 1.1)])

        refractive_indices.append(n)
        densities.append(density_g_cm3)

        refractive_indices_err.append(n / h * h_err)
        densities_err.append(density_err / 1000)

        v_i_bg = bruggeman_vi(epsilon_eff, epsilon_ice)
        v_i_looyenga = looyenga_f(epsilon_eff, eps_air, epsilon_ice)
        v_i_mg = maxwell_garnett_f(epsilon_eff, eps_air, epsilon_ice)

        v_i_bg = np.mean(v_i_bg[(freqs > 0.9) & (freqs < 1.1)])
        v_i_looyenga = np.mean(v_i_looyenga[(freqs > 0.9) & (freqs < 1.1)])
        v_i_mg = np.mean(v_i_mg[(freqs > 0.9) & (freqs < 1.1)])

        v_i_looyengas.append(v_i_looyenga.real * solid_ice_density)
        v_i_bgs.append(v_i_bg.real * solid_ice_density)
        v_i_mgs.append(v_i_mg.real * solid_ice_density)


    axes[0].set_xlabel("Time (ps)")
    axes[0].set_ylabel("Amplitude (a.u.)")

    axes[1].set_xlabel("Frequency (THz)")
    axes[1].set_ylabel("Refractive Index [-]")

    plt.show()

    # plt.style.use('dark_background')

    # plt.scatter(densities, refractive_indices, label="Measured data")
    # plt.errorbar(densities, refractive_indices, xerr=densities_err, yerr=refractive_indices_err, fmt='.', markeredgecolor="black", ecolor='black',
    #              marker="o",
    #              capthick=2, color="red", capsize=2, elinewidth=1, markeredgewidth=0.5, ms=5, label="Measured Data")

    plt.scatter(densities, v_i_bgs, label="Bruggeman")
    plt.scatter(densities, v_i_looyengas, label="Looyenga")
    plt.scatter(densities, v_i_mgs, label="Maxwell-Garnett")

    plt.plot([0,1],[0,1], ls="--", color="gray", label="Ideal")

    plt.xlabel(r"True Ice Density [g/cm$^3$]")
    plt.ylabel("Calculated Ice Density [g/cm$^3$]")
    plt.legend()
    plt.savefig("vol_ratio.png", dpi=300)
    plt.savefig("vol_ratio.pdf")
    plt.show()
