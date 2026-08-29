# THz Ice Porosity Analysis

This project analyzes terahertz time-domain spectroscopy (THz-TDS) measurements of solid and porous ice. It derives
refractive indices, compares solid-ice measurements with literature, estimates porosity with effective-medium theory (
EMT), and visualizes spatially resolved measurements.

## Environment

Install dependencies with:

```bash
uv sync
```

to set up the project.

## Data

The `data/` directory contains campaign configs, raw THz acquisitions, the literature refractive-index table, and
supporting inputs. The `data_template/` directory provides a small template for creating or distributing the same
layout without the full raw dataset. Data can be obtained from Figshare or replaced with acquisitions using the same
structure. See [`data/README.md`](data/README.md) for the complete layout and config schema.

## Project structure

```text
code/          Python analysis code
data/          Configs, raw measurements, and supporting input data
results/       Generated figures, tables, and arrays
data_template/ Empty data layout
```

All commands below are run from the project root. The scripts resolve bundled input and output paths from their own
location, so they can also be launched using absolute paths from another working directory.

## `analyze_measurement_campaigns.py`

This is the shared config loader and campaign-analysis module. It defines the campaign, measurement, and analysis data
structures used by the other scripts; resolves config paths against the project root; reads and averages `.thz` traces;
calculates density and refractive index; and provides the effective-medium model helpers.

When run directly, it analyzes every JSON config in `data/configs/` by default. It creates a time-trace and
refractive-index overview for each campaign, a combined refractive-index-versus-density plot with shared EMT curves, and
`campaign_analysis_summary.csv` in `results/`.

```bash
python code/analyze_measurement_campaigns.py
```

Analyze selected configs or change the output directory with:

```bash
python code/analyze_measurement_campaigns.py \
  --config data/configs/july6_2026.json \
  --output-dir results \
  --show
```

Without `--show`, figures are saved and closed without opening interactive windows. Other active analysis scripts
import this module from `code/`, so it should remain alongside them.

## `analyze_porosity_emt.py`

This is the campaign-level EMT porosity analysis. It loads one or more JSON configs, skips measurements marked with
`"ignore": true`, averages repeated THz traces, derives complex refractive index spectra, and averages the configured
fit band around 1 THz. Measured mass and geometry provide the comparison density and true porosity. Solid measurements
above the density threshold are combined into a shared solid-ice calibration, after which the Bruggemann model is
inverted to estimate porosity.

Run all configured campaigns:

```bash
python code/analyze_porosity_emt.py
```

Run selected campaigns or choose another output directory:

```bash
python code/analyze_porosity_emt.py \
  --config data/configs/july3.5_2026.json \
  --config data/configs/july4_2026.json \
  --output-dir results
```

The script writes `emt_bruggemann_summary.pdf`, `emt_porosity_measurements.csv`, and `emt_porosity_model_metrics.csv`.
Add `--show` to request interactive plot display.

## `compare_solid_ice.py`

This script loads every campaign config, selects non-ignored measurements labeled `SOLID`, derives their
refractive-index spectra, and compares them with `ice_refractive_index.csv` (Tao et al. 2024). It plots each measured
curve plus their mean between 0.8 and 2.0 THz.

```bash
python code/compare_solid_ice.py
```

It writes `solid_ice.png` and `solid_ice.pdf` to `results/` and opens the plot interactively. Both the
literature CSV and output directory are resolved relative to the script location, so the script also works when launched
from another working directory.

## `vacuum_vs_ambient_pressure.py`

This focused comparison uses the July 6 config and its two 5 mm, 5.02 g frost acquisitions. It compares vacuum and
ambient-pressure sample traces, estimates timing drift from each file's embedded `Ref` trace, shifts the ambient sample
trace, and compares the resulting refractive-index spectra.

```bash
python code/vacuum_vs_ambient_pressure.py
```

The configured acquisition paths point into `data/raw`. The script writes `vacuum_vs_ambient_pressure.png` and
`.pdf` to `results/` and opens the figure interactively.

## `refractive_index_image.py`

This is the spatial imaging workflow. It reads a THz image cube and its stored ROI polygons, calculates a
refractive-index spectrum per pixel, averages the 0.9–1.1 THz band into a map, converts that map to porosity with the
Bruggemann EMT model, and reports ROI statistics. A collimated solid-ice/reference pair provides the solid-ice optical
calibration.

The raw image acquisitions, silicon reference, and solid-ice calibration data are included in `data/raw`, and
the script builds all paths relative to its own project directory. Choose an image by changing the module-level
selection constant to one of the keys in `IMAGE_INPUTS`:

```python
SELECTED_IMAGE = "august3_hr_frost"
```

Available keys are `august2_frost`, `august2_hr_frost`, `august3_hr_frost`, `august2_uhr_frost`, and `august2_lr_frost`.
The August 3 high-resolution frost acquisition is selected by default. Before comparing quantitative results, review the
sample thickness, mask center/radius, ROI labels, and true-porosity mass assumptions in the main block for the selected
acquisition.

```bash
python code/refractive_index_image.py
```

The script displays intermediate traces and spectra, prints ROI refractive-index/porosity statistics, and saves the
final PNG and PDF maps to `results/`. The output name includes the selected material label. See [
`data/README.md`](data/README.md#refractive-index-image-data) for the image-acquisition table and
raw-data structure.

## `scicolorscales.py`

This is a supporting data module containing Fabio Crameri scientific color maps converted to Plotly colorscale lists.
Each exported variable, such as `vik`, `acton`, or `lapaz`, is a list of normalized positions and RGB color strings that
can be passed to Plotly as a `colorscale`.

## Config-driven measurement data

The portable JSON configs live in `data/configs/`; all their paths resolve to inputs in `data/raw/`. Generated files
belong in `results/` and are never written below `data/`. See [`data/README.md`](data/README.md) for the complete
directory and schema description.
