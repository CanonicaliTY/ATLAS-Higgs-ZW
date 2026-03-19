# ATLAS Open Data Dilepton Analysis

This directory contains a refactored ATLAS Open Data dilepton analysis for the Manchester 3rd-year lab. The code measures the Z->ll cross section with a cut-and-count method,

`sigma = (N_selected - N_background) / (epsilon * L)`

while keeping the memory-friendly tight-parquet workflow used in the lab notebooks.

## Purpose

The analysis now does five things in one reproducible workflow:

1. builds or reuses channel-specific tight parquet samples,
2. loads OS and SS event samples for plots and cutflows,
3. scans or applies fixed isolation cuts,
4. evaluates an additional data-driven background from control regions,
5. computes the nominal cross section and systematic uncertainties after the chosen additional-background method has been applied.

The primary signal efficiency `epsilon` still comes from `Zee` in the electron channel and `Zmumu` in the muon channel. The low-mass DY samples `m10_40_Zee` and `m10_40_Zmumu` are still plotted and counted as signal-like MC contributions, but they are not used as the primary efficiency sample.

## Directory Structure

```text
ATLAS-parquet/
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

## Quick Start

Run from this directory:

```bash
python3 main.py
```

Use a Python environment that has the scientific stack installed (`awkward`, `numpy`, `pandas`, `matplotlib`, `pyarrow`, `vector`). In this repository the project virtualenv is the safest choice.

The old habit still works:

```bash
python3 RearrangedCode_v2.0.py
```

Outputs are written under:

```text
output_py/run_YYYYMMDD_HHMMSS/<channel>/
```

## Configuration Overview

Edit `config.py`.

The main switches are:

- `SETTINGS["LEPTONS"]`: `("mu",)`, `("e",)`, or `("mu", "e")`
- `SETTINGS["USE_SCAN"]`: run the isolation scan or use the fixed working point
- `SETTINGS["FIXED_ISO"]`: fixed isolation choice when the scan is disabled
- `SETTINGS["ORDER_MODE"]`: how to apply the additional-background estimator
- `SETTINGS["ADDITIONAL_DATA_DRIVEN_BKG"]["METHOD"]`: `wrong_flavour`, `wrong_charge`, `both_average`, or debug `both_sum`
- `SETTINGS["SYSTEMATICS"]["COMBINATION_MODE"]`: `conservative_envelope` or `quadrature`

Recommended nominal settings for the lab:

- `METHOD = "both_average"`
- `ORDER_MODE = "recompute_after_iso"`
- `COMBINATION_MODE = "conservative_envelope"`

## Tight-Parquet Workflow

The code keeps the notebook-style tight-parquet design because the full parquet samples can be too large to hold comfortably in memory.

The workflow is:

1. read the raw parquet samples with a baseline dilepton preselection,
2. write a tighter parquet containing only the fields needed downstream,
3. read the tight parquet back for OS and SS analysis,
4. stream the separate control parquet row-group by row-group when evaluating the control-region study.

This keeps the peak memory low while still letting the final selections, scan, and systematics be recomputed cleanly.

## Control Regions And Additional Background

The additional background estimate uses the residual `Data - MC` in control regions.

The physical regions are:

- `ep_mum`: one `e+` and one `mu-`, independent of index order
- `mup_em`: one `mu+` and one `e-`, independent of index order
- `ee_ss`: same-sign `ee`
- `mumu_ss`: same-sign `mumu`

The old bug came from assuming a fixed lepton index order. The refactored code now treats `ep_mum` and `mup_em` as physical charge-flavour categories, not index-ordered categories.

For debugging, the code saves control-region tables at three stages:

1. baseline control preselection,
2. mass-only selection,
3. mass + iso selection.

Each table includes:

- ordered opposite-flavour counts `(11,13)` and `(13,11)`,
- physical `ep_mum` and `mup_em`,
- same-sign `ee_ss` and `mumu_ss`.

CSV, JSON, and a short text diagnosis are written in `additional_data_driven_bkg/`.

## Order Modes

The estimator is cut-dependent, so the order matters.

`recompute_after_iso`

- apply the final mass + iso working point to the control samples,
- compute the selected estimator from the surviving control-region yields,
- use that background in the quoted sigma.

`fixed_before_iso`

- compute the selected estimator once after baseline + mass-window selection,
- keep that estimate fixed while varying the signal-region isolation cut.

`compare_both`

- the code still uses `recompute_after_iso` as the nominal quoted mode,
- but it also writes the fixed-before-iso comparison table automatically.

The file `additional_data_driven_bkg/<channel>_order_mode_comparison.csv` records the difference directly.

## Systematics

Systematics are evaluated after the selected additional-background method and order mode have been applied.

The current breakdown is:

- isolation neighbour systematic:
  local grid neighbours around the nominal working point,
- isolation allowed-scan envelope:
  saved as a diagnostic, not included in the default total,
- mass-window systematic:
  small Z-window variations around the nominal window,
- estimator-method systematic:
  spread among `wrong_flavour`, `wrong_charge`, and `both_average`,
- estimator-order systematic:
  difference between `recompute_after_iso` and `fixed_before_iso`.

The default total is `conservative_envelope`, meaning the quoted systematic is the largest included component. If you want a more standard independent-component treatment, switch to `quadrature`.

## Main Outputs

Per channel, the most useful files are:

- `*_cutflow_OS.csv`
- `*_cutflow_SS.csv`
- `*_cross_section.json`
- `additional_data_driven_bkg/*_control_region_debug.csv`
- `additional_data_driven_bkg/*_method_comparison.csv`
- `additional_data_driven_bkg/*_order_mode_comparison.csv`
- `iso_scan/*_sigma_scan_allowed_after_additional_bkg.csv`
- `systematics/*_systematic_breakdown.csv`
- `systematics/*_systematics.json`
- `*_summary.txt`

These are intended to be easy to quote directly in a lab logbook.

## Tests

The tests use only small synthetic awkward arrays and do not depend on the full ATLAS datasets.

Run them with:

```bash
python3 -m unittest discover -s tests
```

They cover:

- order-invariant wrong-flavour masks,
- different behaviour between the two order modes,
- neighbour-based isolation systematic output,
- clean module imports and wrapper compatibility.

## Common Debugging Steps

If the code looks wrong, check these first:

1. open `additional_data_driven_bkg/<channel>_control_region_debug_summary.txt` and confirm whether ordered opposite-flavour pairs are asymmetric,
2. compare `*_method_comparison.csv` and `*_order_mode_comparison.csv`,
3. inspect `systematics/*_systematic_breakdown.csv` to see which component dominates,
4. if the scan is enabled, inspect `iso_scan/*_sigma_scan_allowed_after_additional_bkg.csv`,
5. if a control region is empty after final cuts, check whether the baseline and mass-only stages were already sparse.

## Known Caveats

- The electron channel can show `Data > MC` around the Z peak.
- Treat that as a modelling and systematic discussion point for the lab report. Do not silently ignore it.
- `both_sum` is kept for debugging and comparison. It is not included in the default quoted total systematic.
- The mass-window systematic is intentionally restricted to small variations above the 40 GeV threshold region because the primary signal samples are intended for `m_ll > 40 GeV`.
