# FSR / bremsstrahlung additional study

## Files added
- `fsr_study.py` — standalone script for the muon-channel toy FSR recovery study.

## Why this is separate
The existing `main.py` pipeline already does the nominal lab analysis and a lot of diagnostics.
This additional study is narrower and easier to reason about if it lives in one focused script.
That keeps the core lab workflow stable while you test the professor's idea.

## What `fsr_study.py` does
1. Builds a **new dedicated tight parquet** that keeps `lep_eta`, `lep_phi`, and `lep_e`, so you can rebuild muon 4-vectors.
2. Loads **OS dimuon** data + MC.
3. Defines the control sample **fail nominal but pass loose**.
4. Creates toy corrected masses:
   - `mass_fsr_maxcone`
   - `mass_fsr_both`
5. Saves stacked plots before/after the correction.
6. Runs a small sigma scan around the nominal isolation point.

## First run
Open `fsr_study.py` and check `FSR_SETTINGS` at the top.

The most important switches are:
- `FORCE_REBUILD = True` for the **first** run, because this study needs a new parquet.
- `NOMINAL_ISO`
- `LOOSE_ISO`
- `FSR["APPLY_BELOW_MASS"]`

Then run:

```bash
python fsr_study.py
```

## What to look at first
Inside the created `output_fsr/run_.../` directory:

- `summary.txt`
- `plots/between_raw_zoom.png`
- `plots/between_fsr_maxcone_zoom.png`
- `plots/between_fsr_both_zoom.png`
- `sigma_scan/heatmap_raw.png`
- `sigma_scan/heatmap_masscorr_maxcone.png`
- `sigma_scan/heatmap_masscorr_etsub_maxcone.png`
- `between_mass_windows.csv`

## Suggested workflow
1. Reproduce the **fail nominal / pass loose** control plot.
2. Compare raw mass vs corrected mass.
3. If the low-mass shoulder moves upward, that supports the FSR hypothesis.
4. Compare the sigma heatmaps. The most interesting one is usually:
   - `masscorr_etsub_maxcone`

## Caution
This is a **toy** recovery, not a precision FSR reconstruction.
It is meant to answer:

> can the low-mass excess and the isolation dependence be qualitatively explained by near-muon radiated energy?
