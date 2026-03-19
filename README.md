# ATLAS Open Data Dilepton Analysis

This directory contains the refactored Manchester 3rd-year lab analysis for `Z -> ll` using the ATLAS Open Data parquet workflow.

## File Structure

```text
Analysis code/my-ATLAS-code/ATLAS-parquet/
├── main.py
├── RearrangedCode_v2.0.py
├── config.py
├── selections.py
├── parquet_io.py
├── control_regions.py
├── scan.py
├── systematics.py
├── cross_section.py
├── visualisation.py
├── utils.py
├── tests/
└── output_py/
```

The old wrapper is still valid:

```bash
python3 RearrangedCode_v2.0.py
```

The preferred entry point is:

```bash
python3 main.py
```

Use an environment that has the scientific stack installed (`awkward`, `numpy`, `pandas`, `matplotlib`, `pyarrow`, `vector`). In this repository the project virtualenv is the safest choice.

## What The Analysis Does

The analysis keeps the memory-friendly tight-parquet workflow:

1. build or reuse a tight parquet with the baseline preselection,
2. read the tight parquet back for the OS and SS analysis,
3. choose the isolation working point with the 2D optimisation scan or use a fixed point,
4. evaluate the signal region and the control-region DD background with the same cuts,
5. save plots, cutflows, DD tables, and systematic summaries.

The cut-and-count cross section still uses

`sigma = (N_selected - N_background) / (epsilon * L)`

with the main efficiency `epsilon` taken from `Zmumu` in the muon channel or `Zee` in the electron channel. The low-mass DY samples are still present in the plots, but they are labelled explicitly as `LowMassDY ...` in user-facing outputs.

## Control Regions And DD Background

The wrong-flavour bug is fixed by defining the control regions physically, not by lepton index order:

- `ep_mum`: one `e+` and one `mu-`, regardless of whether the electron is `[0]` or `[1]`
- `mup_em`: one `mu+` and one `e-`, regardless of index order
- `ee_ss`: same-sign `ee`
- `mumu_ss`: same-sign `mumu`

For each cut point, the code now evaluates the signal region and control region consistently with the same mass and isolation cuts.

The code always reports four DD rows together:

- `none`
- `wrong_flavour`
- `wrong_charge`
- `both_average`

`both_sum` is intentionally discarded from user-facing logic and outputs.

## Unified Per-Cut Evaluation

`control_regions.py` provides the source-of-truth per-cut evaluation. For one cut point it returns:

- signal-region yields,
- control-region counts and residuals under the same cuts,
- `regions_table`,
- `estimators_table`,
- `sigma_results_table`,
- `comparison_sigma_if_applied`.

This same cut-consistent evaluation is reused for:

- the nominal final cut point,
- mass-window variations,
- scan diagnostics over `(ptcone, etcone)`.

## Isolation Scan

The isolation scan has two separate roles.

### Optimisation Scan

The optimisation scan still searches the 2D `(ptcone, etcone)` grid for the best isolation working point. The scan table now includes an explicit `significance` column and the chosen point is the allowed point with the best significance, subject to the OS signal-efficiency floor.

### Scan Diagnostics / Monotonicity Study

When enabled, the scan diagnostics evaluate the DD-aware sigma values at every scan point and save:

- the full diagnostic scan table,
- significance heatmaps,
- sigma heatmaps for `none`, `wrong_flavour`, `wrong_charge`, and `both_average`,
- optional 3D surfaces,
- sigma slice plots,
- monotonicity delta tables,
- monotonicity classification tables,
- a compact monotonicity summary text file.

These outputs are diagnostic and exploratory. The code does **not** define the official isolation systematic as a max neighbouring-point difference.

## Systematics

The main automatic scalar systematic kept here is the mass-window variation study.

For each mass-window variation, the code reruns the same cut-consistent evaluation and saves:

- sigma values for `none`, `wrong_flavour`, `wrong_charge`, and `both_average`,
- sigma shifts by method,
- a per-method uncertainty summary table.

The code keeps statistical and luminosity uncertainties separate in the saved tables.

The code does **not** include:

- an order-mode systematic,
- a neighbouring-point isolation systematic,
- a forced single “selected” DD method.

## Key Outputs In `output_py/`

For each channel, inspect:

- `summary.txt`
- `cutflow_OS.csv`
- `cutflow_SS.csv`
- `additional_data_driven_bkg/*_regions_table.csv`
- `additional_data_driven_bkg/*_estimators_table.csv`
- `additional_data_driven_bkg/*_sigma_results_table.csv`
- `additional_data_driven_bkg/*_comparison_sigma_if_applied.csv`
- `additional_data_driven_bkg/*_control_region_debug.csv`
- `systematics/*_mass_window_variations.csv`
- `systematics/*_uncertainty_summary_by_method.csv`
- `iso_scan/*_scan_diagnostics_full.csv`
- `iso_scan/*_monotonicity_classification.csv`
- `iso_scan/*_scan_diagnostics_summary.txt`

The most important manual checks after a run are:

1. the `sigma(ptcone, etcone)` heatmaps,
2. the significance heatmap,
3. the monotonicity summary and classification tables,
4. the per-method DD sigma comparison table.

## Tests

The tests use only synthetic awkward arrays and do not require the ATLAS datasets.

Run them with:

```bash
./.venv/bin/python -m unittest discover -s tests
```

They cover:

- the order-invariant wrong-flavour masks,
- the per-cut evaluation method rows,
- absence of `both_sum` and selected-method outputs,
- scan-diagnostic columns,
- monotonicity classification,
- import and wrapper smoke checks.

## Known Caveat

The electron channel can still show `Data > MC` around the Z peak. Treat that as a modelling and systematic discussion point in the lab report rather than silently ignoring it.

