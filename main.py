from pathlib import Path

import numpy as np
from math_utils import get_fft
from numpy.f2py.symbolic import as_expr
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


def average_img(path: Path):
    thz_filees = get_thz_files(path)
    for file in thz_filees:
        # there should be only one...
        with DotthzFile(file) as f:
            t = f["Image"].datasets["time"][:]
            d = f["Image"].datasets["dataset"][:]
            d = np.mean(d, axis=(0, 1))

            d = d[t < 1960]
            t = t[t < 1960]
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
    nx, ny = traces.shape[0], traces.shape[1]
    data_ref = np.array([t_ref, p_ref])

    # Determine frequency axis from the first pixel that yields a valid result
    freq = None
    freq_len = None
    for x in range(nx):
        for y in range(ny):
            trace = traces[x, y, :]
            p_pair = np.array([time, trace])
            try:
                sample, reference = common_window([p_pair, data_ref],
                                                  half_width=window_half_width, win_func=win_func)
                buffer_pixel = uniform_slab(d_mm,
                                            sample, reference, n_med=1,
                                            upsampling=1, min_frequency=min_frequency, max_frequency=max_frequency,
                                            all_optical_constants=True)
                freq = np.array(buffer_pixel["frequency"])
                freq_len = len(freq)
                break
            except Exception:
                # try next pixel
                continue
        if freq is not None:
            break

    if freq is None:
        raise RuntimeError("Unable to compute frequency axis from any pixel. Check input traces/ref.")

    # Preallocate arrays with homogeneous shape and fill with NaN for failures
    refractive_index = np.full((nx, ny, freq_len), np.nan, dtype=float)
    absorption_coefficient = np.full((nx, ny, freq_len), np.nan, dtype=float)
    complex_refractive_index = np.full((nx, ny, freq_len), np.nan + 0j, dtype=complex)

    for x in range(nx):
        for y in range(ny):

            if mask_radius:

                mask_radius_mm = mask_radius * 1 / 0.5

                if ((x - nx // 2) ** 2 + (y - ny // 2) ** 2) > mask_radius_mm ** 2:
                    continue
            trace = traces[x, y, :]
            p_pair = np.array([time, trace])
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
                    complex_refractive_index[x, y, :] = n
                    refractive_index[x, y, :] = n.real
                    absorption_coefficient[x, y, :] = alpha
                else:
                    # leave as NaN if mismatch
                    continue
            except Exception:
                # leave as NaN on any error
                continue

    return freq, refractive_index, complex_refractive_index


if __name__ == '__main__':

    # fig, axes = plt.subplots(nrows=2, figsize=(8, 6))

    solid_ice_path = Path("/Users/linus/Documents/10mm_solid_ice_trans_focus/data_image")

    t_solid, p_solid = load_img(solid_ice_path)

    # p = np.mean(p_solid, axis=(0, 1))
    # f, a, arg = get_fft( t_solid, p)
    #
    # axes[0].semilogy(f, a, label=f"reference solid ice")
    # axes[1].plot(f, arg, label=f"reference solid ice")

    silicon_ref_path = Path("/Users/linus/Documents/silicon_reference_trans_img_porosity/data_image")
    t_ref, p_ref = average_img(silicon_ref_path)

    # f, a, arg = get_fft(t_ref, p_ref)
    #
    # axes[0].semilogy(f, a, label=f"silicon reference")
    # axes[1].plot(f, arg, label=f"silicon reference")

    measurements = [
        # (Path("/Users/linus/Documents/spipa_b_ice_7mm_trans_10g_uniform_1/data_image"), "SPIPA-B", 0.01, 7),
        # (Path("/Users/linus/Documents/spipa_b_ice_7mm_trans_10g_uniform_2/data_image"), "SPIPA-B", 0.01, 7),
        # (Path("/Users/linus/Documents/spipa_b_ice_7mm_trans_13g_uniform_1/data_image"), "SPIPA-B", 0.013, 8),
        # (Path("/Users/linus/Documents/spipa_b_ice_7mm_trans_13g_uniform_2/data_image"), "SPIPA-B", 0.013, 8),
        # (Path("/Users/linus/Documents/spipa_b_ice_10_mm/data_image"), "SPIPA-B", 0.01477, 10),
        (Path("/Users/linus/Documents/spipa_b_ice_10_mm_2/data_image"), "SPIPA-B", 0.01433, 10),
        (Path("/Users/linus/Documents/spipa_b_ice_10_mm_3/data_image"), "SPIPA-B", 0.01436, 10),
        (Path("/Users/linus/Documents/frost_ice_10_mm_3/data_image"), "FROST", 0.00395, 10),
        (Path("/Users/linus/Documents/frost_ice_10_mm_4/data_image"), "FROST", 0.00518, 10)
    ]

    freqs_solid, refractive_index_solid, complex_refractive_index_solid = get_refraction_index(t_solid, p_solid, t_ref,
                                                                                               p_ref,
                                                                                               window_half_width=25,
                                                                                               win_func="hanning",
                                                                                               min_frequency=0.2,
                                                                                               max_frequency=3,
                                                                                               d_mm=10.0,
                                                                                               mask_radius=None)
    n_ice = np.nanmean(complex_refractive_index_solid, axis=(0, 1))
    epsilon_ice = n_ice ** 2
    solid_ice_density = 0.918  # g/cm3

    # axes[0].plot(freqs_solid, np.mean((complex_refractive_index_solid**2).imag, axis=(0,1)), label=f"solid ice")
    # axes[1].plot(freqs_solid, np.mean((complex_refractive_index_solid**2).imag, axis=(0,1)), label=f"solid ice")

    # axes[0].plot(t_solid, np.mean(p_solid, axis=(0,1)), label="Solid ice")
    # axes[1].plot(freqs_solid, np.nanmean(refractive_index_solid, axis=(0, 1)), label="Solid ice")

    for measurement in measurements:
        r = 0.03  # m
        h = measurement[3] / 1000  # m
        volume = np.pi * r ** 2 * h  # m^3
        mass = measurement[2]  # kg
        density = mass / volume  # kg/m^3
        density_g_cm3 = density / 1e3  # g/cm^3
        # print(mass, volume, density, density_g_cm3)

        t, p = load_img(measurement[0])

        freqs, refractive_index, complex_refractive_index = get_refraction_index(t, p, t_ref, p_ref,
                                                                                 window_half_width=30,
                                                                                 win_func="hanning",
                                                                                 min_frequency=0.2, max_frequency=3,
                                                                                 d_mm=measurement[3],
                                                                                 mask_radius=None)

        # axes[0].plot(t, np.mean(p, axis=(0, 1)), label=f"Porous = {density_g_cm3:.3f}")

        # axes[1].plot(freqs, np.nanmean(refractive_index, axis=(0, 1)), label=f"Porous = {density_g_cm3:.3f}")

        neff = np.nanmean(complex_refractive_index, axis=(0, 1))
        epsilon_eff = neff ** 2

        v_i_bg = bruggeman_vi(epsilon_eff, epsilon_ice)
        v_i_looyenga = looyenga_f(epsilon_eff, eps_air, epsilon_ice)
        v_i_mg = maxwell_garnett_f(epsilon_eff, eps_air, epsilon_ice)

        # plt.imshow(np.mean(v_i_bg.real, axis=2))
        # plt.colorbar()
        # plt.title(f"Real porosity: {density_g_cm3 / solid_ice_density:2f}")
        # plt.savefig(f"real_porosity_{density_g_cm3 / solid_ice_density:.3f}.png")
        # plt.show()

        # print(f"Calculated porosity: {np.abs(np.mean(v_i_bg)):2f}")
        # print(f"Real porosity: {density_g_cm3 / solid_ice_density:2f}")
        # print("-----")

        # p = np.mean(p, axis=(0, 1))
        # f, a, arg = get_fft(t, p)
        #
        # axes[0].semilogy(f, a, label=f"{measurement[1]} {density_g_cm3:.3f} g/cm³")
        # axes[1].plot(f, arg, label=f"{measurement[1]} {density_g_cm3:.3f} g/cm³")

        # plt.plot(np.mean(neff.imag, axis=(0,1)), np.mean(neff.real, axis=(0,1)), label=f"{measurement[1]} {density_g_cm3:.3f} g/cm³")
        # axes[0].plot(freqs, np.mean(epsilon_eff.imag, axis=(0,1)), label=f"{measurement[1]} {density_g_cm3:.3f} g/cm³")
        # axes[1].plot(freqs, np.mean(epsilon_eff.real, axis=(0,1)), label=f"{measurement[1]} {density_g_cm3:.3f} g/cm³")

        for v_i, label in [(v_i_bg, "Bruggeman"), (v_i_looyenga, "Looyenga"), (v_i_mg, "Maxwell-Garnett")]:

            print("Model:", label)
            print(f"Calculated porosity: {np.abs(np.mean(v_i)):2f}")
            print(f"Real porosity: {density_g_cm3 / solid_ice_density:2f}")
            print("-----")

            plt.plot(freqs, v_i.real, label=f"{measurement[1]} {density_g_cm3:.3f} g/cm³")
            plt.axhline(density_g_cm3 / solid_ice_density, color='gray', linestyle='--')
            plt.show()

    # plt.legend()
    # plt.show()
