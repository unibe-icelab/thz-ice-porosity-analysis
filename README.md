# THz Ice Porosity Analysis

This project analyzes terahertz time-domain spectroscopy (THz-TDS) measurements of solid and porous ice. It derives refractive indices, compares solid-ice measurements with literature, estimates porosity with effective-medium theory (EMT), and visualizes spatially resolved measurements.

Raw inputs and campaign metadata are stored in [`measurements`](measurements/README.md). Run the commands below from the project root unless noted otherwise.

## Environment

The analysis uses Python plus `numpy`, `matplotlib`, `pandas`, `pydotthz`, and `thzpy`. The imaging workflow additionally imports `cmcrameri`, `plotly`, `shapely`, `pyvista`, and `scikit-image`. The local `thz-analysis` package declares the core THz dependencies and can be installed in editable mode:

```bash
python -m pip install -e thz-analysis
```

Install the remaining plotting and imaging packages in the same environment as required. The repository currently does not provide a single root-level dependency lock file.

## `analyze_porosity_emt.py`

This is the campaign-level EMT porosity analysis. It loads one or more JSON configs, skips measurements marked with `"ignore": true`, averages repeated THz traces, derives complex refractive index spectra, and averages the configured fit band around 1 THz. Measured mass and geometry provide the comparison density and true porosity. Solid measurements above the density threshold are combined into a shared solid-ice calibration, after which the Bruggemann model is inverted to estimate porosity.

Run all configured campaigns:

```bash
python analyze_porosity_emt.py
```

Run selected campaigns or choose another output directory:

```bash
python analyze_porosity_emt.py \
  --config measurements/configs/july3.5_2026.json \
  --config measurements/configs/july4_2026.json \
  --output-dir measurements/output
```

The script writes `emt_bruggemann_summary.pdf`, `emt_porosity_measurements.csv`, and `emt_porosity_model_metrics.csv`. Add `--show` to request interactive plot display.

## `compare_solid_ice.py`

This script loads every campaign config, selects non-ignored measurements labeled `SOLID`, derives their refractive-index spectra, and compares them with `ice_refractive_index.csv` (Tao et al. 2024). It plots each measured curve plus their mean between 0.8 and 2.0 THz.

```bash
python compare_solid_ice.py
```

It writes `solid_ice.png` and `solid_ice.pdf` in the project root and opens the plot interactively. Because the literature CSV is addressed relative to the working directory, run this script from the project root.

## `vacuum_vs_ambient_pressure.py`

This focused comparison uses the July 6 config and its two 5 mm, 5.02 g frost acquisitions. It compares vacuum and ambient-pressure sample traces, estimates timing drift from each file's embedded `Ref` trace, shifts the ambient sample trace, and compares the resulting refractive-index spectra.

```bash
python vacuum_vs_ambient_pressure.py
```

The configured acquisition paths point into `measurements/data`. The script writes `vacuum_vs_ambient_pressure.png` and `.pdf` to `measurements/output/` and opens the figure interactively.

## `refractive_index_image.py`

This is the spatial imaging workflow. It reads a THz image cube and its stored ROI polygons, calculates a refractive-index spectrum per pixel, averages the 0.9–1.1 THz band into a map, converts that map to porosity with the Bruggemann EMT model, and reports ROI statistics. A collimated solid-ice/reference pair provides the solid-ice optical calibration.

The raw image acquisitions, silicon reference, and solid-ice calibration data are included in `measurements/data`, and the script builds all paths relative to its own project directory. Choose an image by changing the module-level selection constant to one of the keys in `IMAGE_INPUTS`:

```python
SELECTED_IMAGE = "august3_hr_frost"
```

Available keys are `august2_frost`, `august2_hr_frost`, `august3_hr_frost`, `august2_uhr_frost`, and `august2_lr_frost`. The August 3 high-resolution frost acquisition is selected by default. Before comparing quantitative results, review the sample thickness, mask center/radius, ROI labels, and true-porosity mass assumptions in the main block for the selected acquisition.

```bash
python refractive_index_image.py
```

The script displays intermediate traces and spectra, prints ROI refractive-index/porosity statistics, and saves the final map in PNG and PDF form in the current working directory. The output name includes the selected material label. See [`measurements/README.md`](measurements/README.md#refractive-index-image-data) for the image-acquisition table and raw-data structure.

## Config-driven measurement data

The portable JSON configs live in `measurements/configs/`; all their paths resolve to copied inputs in `measurements/data/`. See [`measurements/README.md`](measurements/README.md) for the complete directory and schema description.
