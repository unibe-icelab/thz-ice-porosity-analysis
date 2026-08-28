# Measurements

This directory contains the THz-TDS measurement inputs, campaign configuration files, and generated analysis products
used by the scripts in the project root.

## Directory structure

```text
measurements/
├── configs/                 Campaign descriptions and analysis settings (JSON)
├── data/                    Copied measurement acquisitions
│   └── <acquisition>/
│       ├── data/
│       │   ├── single_pixel/
│       │   └── trans/single_pixel/
│       └── data_image/
└── output/                  Generated figures, tables, and NumPy arrays
```

Each directory below `data/` keeps the original acquisition name. The JSON configs point to a `single_pixel` directory
containing one or more `.thz` files. Some acquisitions also contain `.png` and `.pdf` exports produced by the
measurement software; these are retained alongside the raw `.thz` files.

The two layouts in use are:

- `data/<acquisition>/data/single_pixel/` for the legacy collimated measurements.
- `data/<acquisition>/data/trans/single_pixel/` for the April and July transmission campaigns.
- `data/<acquisition>/data_image/` for spatially resolved image cubes used by `refractive_index_image.py`.

## Refractive-index image data

The raw inputs referenced by `refractive_index_image.py` are included in `data/`. The default selection is the August 3
high-resolution frost image. Alternative image resolutions can be selected with the `SELECTED_IMAGE` constant in that
script.

| Selection key       | Acquisition                                           | Material label |
|---------------------|-------------------------------------------------------|----------------|
| `august2_frost`     | `porosity_august2_2026_focused_frost_5.0mm_1.7g`      | Frost          |
| `august2_hr_frost`  | `porosity_august2_2026_hr_focused_frost_10.0mm_7.2g`  | Frost HR       |
| `august3_hr_frost`  | `porosity_august3_hr_2026_focused_frost_10.0mm_7.75g` | Frost HR       |
| `august2_uhr_frost` | `porosity_august2_uhr_2026_focused_frost_10.0mm_7.2g` | Frost UHR      |
| `august2_lr_frost`  | `porosity_august2_lr_2026_focused_frost_10.0mm_7.2g`  | Frost LR       |

The image workflow also uses:

- `porosity_august3_2026_focused_silicon_reference/data/trans/single_pixel/` as its silicon reference.
- `collimated_solid_ice_3a/data/single_pixel/` as its solid-ice calibration sample.
- `collimated_silicon_metal_sheet_fix_focus/data/single_pixel/` as the solid-ice reference.

Image acquisitions store a spatial THz trace cube and ROI metadata in a `.thz` file. The script reads the image axes and
pixel spacing from the measurement metadata, calculates a spectrum for each pixel, and uses the stored ROI polygons when
reporting statistics. The accompanying PNG and PDF files are retained as acquisition exports where present.

Paths stored in the configs are relative to the project root, for example:

```json
"path": "measurements/data/porosity_july6_2026_frost_5.0mm_5.02g/data/trans/single_pixel"
```

`analyze_measurement_campaigns.py` resolves these paths against the project root, so analysis commands can be started
from another working directory as well.

## Campaign config files

Every JSON file in `configs/` describes one campaign. The top-level fields are:

- `campaign_id`: name written to output tables and used in plot labels.
- `description`: short description of the campaign and reference setup.
- `plot_color`: Matplotlib color used for the campaign.
- `analysis`: shared geometry, uncertainty, windowing, frequency, fit-band, and solid-ice calibration settings.
- `reference`: default reference acquisition for the campaign.
- `measurements`: sample measurements and their physical properties.

The `analysis` object contains:

- `radius_m` and `radius_err_m`: sample-holder radius and uncertainty.
- `thickness_err_mm` and `mass_err_kg`: campaign defaults for uncertainty calculations.
- `window_half_width` and `window_function`: time-domain window settings.
- `min_frequency_thz` and `max_frequency_thz`: frequency range passed to the slab inversion.
- `fit_band_thz`: frequency interval averaged for the reported refractive index near 1 THz.
- `solid_ice_window_half_width`, `solid_ice_thickness_mm`, and `solid_ice_density_g_cm3`: solid-ice calibration
  settings.
- `solid_ice_measurement_path` and `solid_ice_reference_path`: sample and reference inputs used for that calibration.

Each item in `measurements` contains:

- `measurement_id`: identifier for tables and diagnostics.
- `label`: material class such as `SOLID`, `SPIPA-B`, or `FROST`.
- `path`: directory containing the sample `.thz` files.
- `reference_path`: optional per-measurement reference override; the campaign-level reference is used when omitted.
- `mass_kg`, `mass_err_kg`, `thickness_mm`, and `thickness_err_mm`: sample properties and uncertainties.
- `ignore`: when `true`, campaign analyses skip the measurement while retaining its metadata.
    - files are ignored, if there have been issues during the measurement, often visible either in frequency domain when
      looking at the refractive index. This is likely due to a misalignment or errors in sample preparation.

Note: the `.thz` raw files also contain mass and thickness values, but the more accurate ones are in the `config` files.
To analyze all configs, run from the project root:

```bash
python analyze_measurement_campaigns.py
```

To select one config:

```bash
python analyze_measurement_campaigns.py --config measurements/configs/july6_2026.json
```

Outputs are written to `measurements/output/` by default and can be displayed with the `--show` argument.
