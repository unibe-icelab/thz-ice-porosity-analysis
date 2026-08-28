# python
import ast
import os
from pathlib import Path
from typing import Dict, Optional

from cmcrameri import cm
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon as MplPolygon
from plotly.subplots import make_subplots
from pydotthz import DotthzFile
from shapely.geometry import Point, Polygon
from thzpy.timedomain import common_window
from thzpy.transferfunctions import uniform_slab
from scicolorscales import vik

import pyvista as pv
import numpy as np
from skimage.measure import marching_cubes

EPS_AIR = 1.0
SOLID_ICE_DENSITY_G_CM3 = 0.918
DEFAULT_EMT_MODEL = "bruggemann"

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


def get_thz_file_from_path(path: Path):
    for file in os.listdir(path):
        if file.endswith("_data.thz"):
            return path.joinpath(file)


def extract_rois(path: Path, measurement_key: Optional[str] = None) -> Dict[str, Dict[str, object]]:
    rois: Dict[str, Dict[str, object]] = {}
    with DotthzFile(path, "r") as image_file:
        if measurement_key is None:
            measurement_key = list(image_file.get_measurements().keys())[0]

        metadata = image_file[measurement_key].metadata
        height = int(float(metadata["height"]))
        width = int(float(metadata["width"]))

        roi_labels = [label.strip() for label in metadata["ROI Labels"].split(",") if label.strip()]
        for index, roi_label in enumerate(roi_labels):
            roi_raw = metadata[f"ROI {index}"]
            roi_points = ast.literal_eval(roi_raw) if isinstance(roi_raw, str) else roi_raw
            roi_points = list(roi_points)

            roi_points_corrected = [(x, width - 1 - y) for x, y in roi_points]
            polygon = Polygon(roi_points_corrected)
            pixels_inside_roi = [
                (x, y) for y in range(height) for x in range(width) if polygon.contains(Point(x, y))
            ]
            rois[roi_label] = {
                "pixels": pixels_inside_roi,
                "polygon_pixels": roi_points_corrected,
            }

    if not rois:
        raise RuntimeError(f"No ROIs found in {path}")

    return rois


def roi_polygon_mm(points_px, height: int, dx: float, dy: float):
    return [(x * dx, (height - 1 - y) * dy) for x, y in points_px]


def roi_stats(image: np.ndarray, pixels) -> tuple[float, float]:
    values = np.array([image[y, x] for x, y in pixels], dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    return float(np.nanmean(values)), float(np.nanstd(values, ddof=1)) if values.size > 1 else 0.0


def get_refraction_index(time, traces, t_ref, p_ref, window_half_width=15, win_func="hanning",
                         min_frequency=0.2, max_frequency=3, d_mm=1.0, mask_radius=None, mask_center_x=None,
                         mask_center_y=None):
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
                if not mask_center_y:
                    mask_center_y = 0
                if not mask_center_x:
                    mask_center_x = 0

                mask_radius_mm = mask_radius * 1 / 0.5

                if ((x - mask_center_x - nx // 2) ** 2 + (y - mask_center_y - ny // 2) ** 2) > mask_radius_mm ** 2:
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


def maxwell_garnett_vi(epsilon_eff, epsilon_host, epsilon_ice):
    numerator = (epsilon_eff - epsilon_host) * (epsilon_ice + 2 * epsilon_host)
    denominator = (epsilon_ice - epsilon_host) * (epsilon_eff + 2 * epsilon_host)
    return numerator / denominator


def looyenga_vi(epsilon_eff, epsilon_host, epsilon_ice):
    return (np.power(epsilon_eff, 1 / 3) - np.power(epsilon_host, 1 / 3)) / (
            np.power(epsilon_ice, 1 / 3) - np.power(epsilon_host, 1 / 3)
    )


def bruggemann_vi(epsilon_eff, epsilon_ice):
    lhs = (EPS_AIR - epsilon_eff) / (EPS_AIR + 2 * epsilon_eff)
    rhs = (epsilon_ice - epsilon_eff) / (epsilon_ice + 2 * epsilon_eff)
    return lhs / (lhs - rhs)


def porosity_from_emt_refractive_index(n_eff_map, n_ice_scalar, model=DEFAULT_EMT_MODEL):
    epsilon_eff = np.asarray(n_eff_map, dtype=float) ** 2
    epsilon_ice = float(n_ice_scalar) ** 2

    if model == "bruggemann":
        ice_fraction = bruggemann_vi(epsilon_eff, epsilon_ice)
    elif model == "maxwellgarnett":
        ice_fraction = maxwell_garnett_vi(epsilon_eff, EPS_AIR, epsilon_ice)
    elif model == "lll":
        ice_fraction = looyenga_vi(epsilon_eff, EPS_AIR, epsilon_ice)
    else:
        raise ValueError(f"Unsupported EMT model: {model}")

    ice_fraction = np.clip(np.real(ice_fraction), 0.0, 1.0)
    return 1.0 - ice_fraction


# plt.style.use('dark_background')


if __name__ == "__main__":
    # ref_path = Path("/Users/linus/Documents/comet_dust_pellet_HESSO_silicon_ref_vac_3/data_image")
    #
    # path = get_thz_file_from_path(ref_path)
    #
    # with DotthzFile(path) as psf_data:
    #     key = list(psf_data.keys())[0]
    #     datasets = psf_data[key].datasets
    #
    #     # from the first dataset, extract the image:
    #     t_ref = np.array(datasets["time"])
    #     traces_ref = np.array(datasets["dataset"])
    #
    # p_ref = np.mean(traces_ref, axis=(0, 1))

    ref_path = Path(
        "/Users/linus/Documents/porosity_august3_2026_focused_silicon_reference/data/trans/single_pixel/1787033231.1361303_sp_data.thz")
    with DotthzFile(ref_path) as ref_file:
        sample = ref_file["Single Pixel 0"].datasets["Sample"][:]
        t_ref = sample[:, 0]
        p_ref = sample[:, 1]

    solid_ice_path = Path("/Users/linus/Documents/collimated_solid_ice_3a/data/single_pixel")
    solid_ice_ref_path = Path("/Users/linus/Documents/collimated_silicon_metal_sheet_fix_focus/data/single_pixel")
    solid_ice_thz_path = get_thz_file_from_path(solid_ice_path)
    solid_ice_ref_thz_path = get_thz_file_from_path(solid_ice_ref_path)
    with DotthzFile(solid_ice_thz_path) as solid_file:
        solid_sample = solid_file["Single Pixel 0"].datasets["Sample"][:]
        t_solid = solid_sample[:, 0]
        p_solid = solid_sample[:, 1]
    with DotthzFile(solid_ice_ref_thz_path) as solid_ref_file:
        solid_ref_sample = solid_ref_file["Single Pixel 0"].datasets["Sample"][:]
        t_solid_ref = solid_ref_sample[:, 0]
        p_solid_ref = solid_ref_sample[:, 1]

    # SPIPA-B ICE, low res, 38 deg phase angle
    material, path = ("SPIPA-B", Path("/Users/linus/Documents/ICE_PEBBLE_CoDA_T3_dust_low_res_vac_trans_4/data_image"))

    material, path = ("Frost", Path("/Users/linus/Documents/porosity_august2_2026_focused_frost_5.0mm_1.7g/data_image"))
    material, path = ("Frost HR",
                      Path("/Users/linus/Documents/porosity_august2_2026_hr_focused_frost_10.0mm_7.2g/data_image"))
    material, path = ("Frost HR",
                      Path("/Users/linus/Documents/porosity_august3_hr_2026_focused_frost_10.0mm_7.75g/data_image"))
    # material, path = ("Frost UHR", Path("/Users/linus/Documents/porosity_august2_uhr_2026_focused_frost_10.0mm_7.2g/data_image"))
    # material, path = ("Frost LR", Path("/Users/linus/Documents/porosity_august2_lr_2026_focused_frost_10.0mm_7.2g/data_image"))

    # solid ICE, low res, 38 deg phase angle
    # material, path = ("solid", Path("/Users/linus/Documents/ICE_PEBBLE_CoDA_T3_dust_low_res_vac_trans_5/data_image"))

    path = get_thz_file_from_path(path)

    with DotthzFile(path) as psf_data:
        key = list(psf_data.keys())[0]
        dx = float(psf_data[key].metadata['dx [mm]'])
        dy = float(psf_data[key].metadata['dy [mm]'])
        x_min = float(psf_data[key].metadata['x_min [mm]'])
        x_max = float(psf_data[key].metadata['x_max [mm]'])
        y_min = float(psf_data[key].metadata['y_min [mm]'])
        y_max = float(psf_data[key].metadata['y_max [mm]'])
        width = int(float(psf_data[key].metadata['width']))
        height = int(float(psf_data[key].metadata['height']))
        datasets = psf_data[key].datasets

        print(psf_data[key].metadata["description"])

        # from the first dataset, extract the image:
        times = np.array(datasets["time"])
        traces = np.array(datasets["dataset"])
    rois = extract_rois(path, measurement_key=key)

    # traces = traces[::, ::, times < 1960]
    # times = times[times < 1960]

    plt.plot(times, traces[width // 2, height // 2, :])
    plt.plot(times, traces[width // 4, height // 4, :])
    # plt.plot(times, traces[48, 68, :])
    # plt.plot(times, traces[42, 60, :])
    plt.plot(t_ref, p_ref, color="black")
    plt.title("Example Trace Before Surface Extraction")
    plt.xlabel("Time (ps)")
    plt.ylabel("Signal (a.u.)")
    plt.show()

    freqs_solid, _, solid_complex_refractive_index = get_refraction_index(
        t_solid,
        p_solid[np.newaxis, np.newaxis, :],
        t_solid_ref,
        p_solid_ref,
        window_half_width=25,
        win_func="hanning",
        min_frequency=0.5,
        max_frequency=3,
        d_mm=10.0,
    )
    solid_band_mask = (freqs_solid > 0.9) & (freqs_solid < 1.1)
    solid_refractive_index_at_1thz = float(np.nanmean(np.real(solid_complex_refractive_index[0, 0, solid_band_mask])))

    freqs, refractive_index, complex_refractive_index = get_refraction_index(
        times,
        traces,
        t_ref,
        p_ref,
        window_half_width=25,
        win_func="hanning",
        min_frequency=0.5,
        max_frequency=3,
        d_mm=10.0,
        mask_radius=30,
        mask_center_x=-3,
        mask_center_y=3,
    )

    plt.plot(freqs, refractive_index[width // 2, height // 2, :])
    plt.show()

    band_mask = (freqs > 0.9) & (freqs < 1.1)
    refractive_index_image = np.nanmean(refractive_index[:, :, band_mask], axis=2)
    refractive_index_image[refractive_index_image > 2.1] = np.nan

    porosity_image = porosity_from_emt_refractive_index(
        refractive_index_image,
        solid_refractive_index_at_1thz,
        model=DEFAULT_EMT_MODEL,
    )
    porosity_image[np.isnan(refractive_index_image)] = np.nan

    print("ROI statistics")
    for roi_label, roi_data in rois.items():
        n_mean, n_std = roi_stats(refractive_index_image, roi_data["pixels"])
        por_mean, por_std = roi_stats(porosity_image, roi_data["pixels"])
        print(
            f"{roi_label}: "
            f"n = {n_mean:.4f} +/- {n_std:.4f}, "
            f"porosity = {por_mean:.2f} +/- {por_std:.2f}"
        )

    ### true porosity

    print("True Porosity: ")
    A = 604.102  # mm^2
    d = 10 # mm
    d_err = 0.1  # mm
    V = A * d  # mm^3
    m = 1.2  # g
    m_err = 0.25  # g
    for i in range(4):
        mass = m + 0.5 * i  # g
        rho = mass / V * 1000  # g/cm^3
        rho_err = np.sqrt((1 / V * 1000 * m_err ) ** 2 + (mass / (A * d ** 2) * 1000 * d_err) ** 2)  # g/cm^3

        print(
            f"ROI {i + 1}: "
            f"rho = {rho:.4f} +/- {rho_err:.4f}, "
            f"porosity = {1 - rho / SOLID_ICE_DENSITY_G_CM3:.2f} +/- {rho_err / SOLID_ICE_DENSITY_G_CM3:.2f}"
        )

    fig, axes = plt.subplots(ncols=2, figsize=(12, 5), constrained_layout=True)

    n_norm = Normalize(vmin=1.0, vmax=1.5)
    n_im = axes[0].imshow(
        refractive_index_image,
        cmap=cm.lipari_r,
        norm=n_norm,
        aspect="equal",
        extent=(0, x_max - x_min, 0, y_max - y_min),
    )
    axes[0].text(40, 45, "A", color="white")
    axes[0].text(18, 41, "B", color="white")
    axes[0].text(23, 17, "C", color="white")
    axes[0].text(46, 21, "D", color="white")
    # for roi_data in rois.values():
    #     axes[0].add_patch(
    #         MplPolygon(
    #             roi_polygon_mm(roi_data["polygon_pixels"], height, dx, dy),
    #             closed=True,
    #             fill=False,
    #             edgecolor="white",
    #             linewidth=1.5,
    #         )
    #     )

    fig.colorbar(n_im, ax=axes[0], label="Refractive Index @ 1 THz")
    axes[0].set_title("a) Refractive Index Map")
    axes[0].set_xlabel("X [mm]")
    axes[0].set_ylabel("Y [mm]")
    axes[0].set_xticks(range(0, int(x_max - x_min + 1), 10))
    axes[0].set_yticks(range(0, int(y_max - y_min + 1), 10))

    porosity_im = axes[1].imshow(
        porosity_image,
        cmap="Blues_r",
        norm=Normalize(vmin=0.45, vmax=1.0),
        aspect="equal",
        extent=(0, x_max - x_min, 0, y_max - y_min),
    )

    axes[1].text(40, 45, "A", color="white")
    axes[1].text(18, 41, "B", color="white")
    axes[1].text(23, 17, "C", color="white")
    axes[1].text(46, 21, "D", color="white")
    # for roi_data in rois.values():
    #     axes[1].add_patch(
    #         MplPolygon(
    #             roi_polygon_mm(roi_data["polygon_pixels"], height, dx, dy),
    #             closed=True,
    #             fill=False,
    #             edgecolor="white",
    #             linewidth=1.5,
    #         )
    #     )
    fig.colorbar(porosity_im, ax=axes[1], label=r"Porosity")
    axes[1].set_title("b) Porosity Map")
    axes[1].set_xlabel("X [mm]")
    axes[1].set_ylabel("Y [mm]")
    axes[1].set_xticks(range(0, int(x_max - x_min + 1), 10))
    axes[1].set_yticks(range(0, int(y_max - y_min + 1), 10))

    fig.savefig(f"frost_transmission_{material}_inv_and_porosity.png", dpi=300)
    fig.savefig(f"frost_transmission_{material}_inv_and_porosity.pdf")
    plt.show()
