#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ATLAS Open Data (Manchester 3rd year lab): Z → ℓℓ dilepton analysis runner

This is a .py version of the "Rearranged Code" section in DilepLabNotebook.ipynb,
designed for *your own* use (readability > cleverness).

What it does (per chosen lepton channel):
  1) Load OS and SS samples with a baseline 2-lepton selection + trigger + pT cuts.
  2) Plot BEFORE-cuts distributions:
       - leading lepton pT (OS & SS)
       - dilepton invariant mass m_ll (OS & SS)
  3) Define selection utilities (mass window + isolation cuts).
  4) Scan isolation cuts (ptcone + etcone) inside the mass window to pick a "best" working point.
  5) Plot AFTER-cuts m_ll (OS & SS) in the Z peak region.
  6) Produce cutflow tables (OS & SS).
  7) Compute σ(pp→Z→ℓℓ) with stat/syst/lumi uncertainties (cut-and-count).

Notes:
  - Uses the provided backend functions (analysis_parquet, plot_stacked_hist, produced_event_count, etc.).
  - Saves all plots/tables to disk under OUTPUT_DIR.
  - Creates *grouped* figures (2x2, 1x2 grids) by composing the individual saved plots into subplots.
    (This avoids needing an 'ax=' argument in backend plotting.)

Lab constants (from the lab script):
  - Integrated luminosity for provided datasets: 30.6 fb^-1
  - Relative luminosity uncertainty: 1.7%
"""

from __future__ import annotations

# ============================================================
# 0) SETTINGS (edit here only) — no user input at runtime
# ============================================================

SETTINGS = {
    # Run which channel(s)?
    #   ("mu",) for muons, ("e",) for electrons, ("mu","e") for both
    "LEPTONS": ("mu",),

    # Data loading
    "FRACTION": 0.02,      # fraction of parquet files to process (<=1)
    "PT_MIN": 10.0,        # baseline lepton pT threshold (GeV)

    # "Final" physics selection (mass window always applied in the scan)
    "MASS_WINDOW": (66.0, 116.0),   # GeV
    "REQUIRE_BOTH_ISO": True,       # if False: apply isolation only to leading lepton

    # Isolation scan (search region + step)
    "ISO_SCAN": {
        "PTCONE_RANGE": (0.0, 10.0),   # GeV
        "PTCONE_STEP":  0.25,          # GeV
        "ETCONE_RANGE": (0.0, 10.0),   # GeV
        "ETCONE_STEP":  0.25,          # GeV
        "OS_SIG_EFF_MIN": 0.995,       # keep at least this much OS signal (relative to mass-only)
    },

    # Plot styling & ranges
    "PLOTS": {
        "LEADING_PT": {"xmin": 0, "xmax": 200, "bins": 50, "logy": True},
        "MASS_FULL":  {"xmin": 0, "xmax": 200, "bins": 120, "logy": True},
        "MASS_ZOOM":  {"xmin": 60, "xmax": 120, "bins": 60, "logy": True},
    },

    # Output
    "OUTPUT_DIR": "output_py",
    "SAVE_PLOTS": True,
    "SAVE_TABLES": True,
    "SAVE_SCAN_TABLES": True,
    "SAVE_JSON": True,
    "MAKE_GROUP_FIGURES": True,  # compose the individual plots into 2x2 / 1x2 grids

    # Environment setup (keep the notebook install logic, but make it optional)
    # Turn these ON if you run from a fresh environment.
    "AUTO_INSTALL": False,             # pip install atlasopenmagic / pinned pyarrow if missing
    "RUN_INSTALL_FROM_ENV_YML": False, # atlasopenmagic.install_from_environment("../backend/environment.yml")

    # If you want to *force* using a fixed isolation cut (no scan), set USE_SCAN=False
    "USE_SCAN": True,
    "FIXED_ISO": {"ptcone_max": 4.5, "etcone_max": 9.25},  # used only if USE_SCAN=False
}


# ============================================================
# 1) "Install" block (converted from notebook cell 1)
# ============================================================

import os
import sys
import subprocess
from pathlib import Path
import datetime

def _pip_install(requirement: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", requirement])

def _ensure_environment() -> None:
    if not SETTINGS["AUTO_INSTALL"] and not SETTINGS["RUN_INSTALL_FROM_ENV_YML"]:
        return

    # atlasopenmagic
    try:
        import atlasopenmagic  # noqa: F401
    except Exception:
        if SETTINGS["AUTO_INSTALL"]:
            _pip_install("atlasopenmagic")
        else:
            raise

    # pyarrow pin (only if requested)
    try:
        import pyarrow  # noqa: F401
        if SETTINGS["AUTO_INSTALL"] and getattr(pyarrow, "__version__", "") != "20.0.0":
            _pip_install("pyarrow==20.0.0")
    except Exception:
        if SETTINGS["AUTO_INSTALL"]:
            _pip_install("pyarrow==20.0.0")
        else:
            raise

    # environment.yml installer (optional)
    if SETTINGS["RUN_INSTALL_FROM_ENV_YML"]:
        from atlasopenmagic import install_from_environment
        env_file = (Path(__file__).resolve().parent / "../backend/environment.yml").resolve()
        install_from_environment(environment_file=str(env_file))


# ============================================================
# 2) Imports block (converted from notebook cell 2)
# ============================================================

# Make matplotlib safe for non-interactive script usage
import matplotlib
matplotlib.use("Agg")  # always save to files; comment out if you prefer interactive display

import re
import time
import glob
import numpy as np
import pandas as pd
import awkward as ak
import vector
import hist  # noqa: F401
from hist import Hist  # noqa: F401
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator  # noqa: F401
import pyarrow.parquet as pq  # noqa: F401


def _import_backend():
    """
    Add the parent directory to sys.path so `import backend` works
    when this file is placed in `.../ATLAS-parquet/`.
    """
    here = Path(__file__).resolve()
    parent = here.parent.parent  # `.../my-ATLAS-code/`
    if str(parent) not in sys.path:
        sys.path.append(str(parent))

    from backend import (
        get_valid_variables, validate_read_variables, VALID_STR_CODE,  # noqa: F401
        plot_stacked_hist, plot_histograms, histogram_2d, plot_errorbars,  # noqa: F401
        get_histogram, analysis_parquet, produced_event_count
    )
    return {
        "get_valid_variables": get_valid_variables,
        "validate_read_variables": validate_read_variables,
        "analysis_parquet": analysis_parquet,
        "plot_stacked_hist": plot_stacked_hist,
        "produced_event_count": produced_event_count,
    }


# ============================================================
# 3) Analysis configuration (channel-dependent)
# ============================================================

CHANNELS = {
    "mu": {
        "string_codes": ["2to4lep", "Zmumu", "m10_40_Zmumu", "Ztautau", "ttbar", "Wmunu"],
        "type_sum": 26,                 # 13 + 13
        "trigger_field": "trigM",
        "signals": ["Zmumu", "m10_40_Zmumu"],
        "bkgs":    ["Ztautau", "ttbar", "Wmunu"],
        "leading_label": r"leading muon $p_T$ [GeV]",
    },
    "e": {
        "string_codes": ["2to4lep", "Zee", "m10_40_Zee", "Ztautau", "ttbar", "Wenu"],
        "type_sum": 22,                 # 11 + 11
        "trigger_field": "trigE",
        "signals": ["Zee", "m10_40_Zee"],
        "bkgs":    ["Ztautau", "ttbar", "Wenu"],
        "leading_label": r"leading electron $p_T$ [GeV]",
    }
}

BASE_VARS = [
    "lep_n", "lep_pt", "lep_eta", "lep_phi", "lep_e",
    "lep_type", "lep_charge",
    "trigM", "trigE",
]

ISO_VARS = [
    "lep_ptvarcone30",     # track isolation (ptcone)
    "lep_topoetcone20",    # calo isolation (etcone)
]


# ============================================================
# 4) Core selection / utilities (from your rearranged notebook)
# ============================================================

def base_dilepton(data: ak.Array, lepton: str, pt_min: float) -> ak.Array:
    """Baseline: exactly 2 leptons, correct flavour, pT thresholds, trigger, compute mass."""
    cfg = CHANNELS[lepton]

    data = data[data["lep_n"] == 2]
    data = data[(data["lep_type"][:, 0] + data["lep_type"][:, 1]) == cfg["type_sum"]]
    data = data[(data["lep_pt"][:, 0] > pt_min) & (data["lep_pt"][:, 1] > pt_min)]
    data = data[data[cfg["trigger_field"]]]

    p4 = vector.zip({
        "pt":  data["lep_pt"],
        "eta": data["lep_eta"],
        "phi": data["lep_phi"],
        "E":   data["lep_e"],
    })
    data["mass"] = (p4[:, 0] + p4[:, 1]).M
    return data


def make_cut_function(lepton: str, sign: str, pt_min: float):
    """Return a cut_function(data)->data suitable for analysis_parquet()."""
    sign = sign.upper().strip()

    def _cut(data: ak.Array) -> ak.Array:
        d = base_dilepton(data, lepton=lepton, pt_min=pt_min)
        qprod = d["lep_charge"][:, 0] * d["lep_charge"][:, 1]
        if sign == "OS":
            return d[qprod == -1]
        if sign == "SS":
            return d[qprod == +1]
        raise ValueError("sign must be 'OS' or 'SS'")

    return _cut


def infer_suffix(d: dict, data_code: str = "2to4lep") -> str:
    """Infer the suffix like '0_02' from keys such as '2to4lep_0_02'."""
    for k in d.keys():
        if k.startswith(data_code + "_"):
            return k.split(data_code + "_", 1)[1]
    raise KeyError(f"Cannot infer suffix: no {data_code}_* key found.")


def build_plot_dict(d: dict, suffix: str, lepton: str) -> dict:
    """
    Create plot_dict with stable key ordering:
      Data, Signal*, Background*
    """
    cfg = CHANNELS[lepton]
    out = {"Data": d.get(f"2to4lep_{suffix}")}

    for s in cfg["signals"]:
        out[f"Signal {s}"] = d.get(f"{s}_{suffix}")
    for b in cfg["bkgs"]:
        out[f"Background {b}"] = d.get(f"{b}_{suffix}")
    return out


def _weight_field(events: ak.Array | None) -> str | None:
    if events is None:
        return None
    for cand in ("weight", "totalWeight"):
        if cand in events.fields:
            return cand
    return None


def yield_data(events: ak.Array | None) -> float:
    return 0.0 if events is None else float(len(events))


def yield_mc(events: ak.Array | None) -> float:
    if events is None:
        return 0.0
    wf = _weight_field(events)
    if wf is None:
        return float(len(events))
    return float(ak.sum(events[wf]))


def yield_mc_var(events: ak.Array | None) -> float:
    """Variance of a weighted yield (sum w^2)."""
    if events is None:
        return 0.0
    wf = _weight_field(events)
    if wf is None:
        return float(len(events))
    w = events[wf]
    return float(ak.sum(w * w))


def apply_selection(events: ak.Array | None,
                    mass_window: tuple[float, float] | None = None,
                    ptcone_max: float | None = None,
                    etcone_max: float | None = None,
                    require_both: bool = True) -> ak.Array | None:
    """
    Apply mass-window and isolation cuts to an event array.
    Important: isolation variables are vectors; we cut on [0] and [1] explicitly.
    """
    if events is None:
        return None

    mask = ak.Array(np.ones(len(events), dtype=bool))

    if mass_window is not None:
        lo, hi = mass_window
        mask = mask & (events["mass"] > lo) & (events["mass"] < hi)

    if ptcone_max is not None:
        if require_both:
            mask = mask & (events["lep_ptvarcone30"][:, 0] < ptcone_max) & (events["lep_ptvarcone30"][:, 1] < ptcone_max)
        else:
            mask = mask & (events["lep_ptvarcone30"][:, 0] < ptcone_max)

    if etcone_max is not None:
        if require_both:
            mask = mask & (events["lep_topoetcone20"][:, 0] < etcone_max) & (events["lep_topoetcone20"][:, 1] < etcone_max)
        else:
            mask = mask & (events["lep_topoetcone20"][:, 0] < etcone_max)

    return events[mask]


def select_plotdict(plot_dict: dict, **kwargs) -> dict:
    return {k: apply_selection(v, **kwargs) if v is not None else None for k, v in plot_dict.items()}


def summarize(plot_dict: dict) -> tuple[float, float, float, float, float]:
    """
    Returns: (N_data, N_sig, N_bkg, S/B, Data/MC)
    Signal/background yields are MC-weighted.
    """
    n_data = yield_data(plot_dict.get("Data"))
    n_sig = 0.0
    n_bkg = 0.0
    for k, v in plot_dict.items():
        if k.startswith("Signal"):
            n_sig += yield_mc(v)
        if k.startswith("Background"):
            n_bkg += yield_mc(v)
    n_mc = n_sig + n_bkg
    sb = (n_sig / n_bkg) if n_bkg > 0 else float("inf")
    dmc = (n_data / n_mc) if n_mc > 0 else float("nan")
    return n_data, n_sig, n_bkg, sb, dmc


def cutflow_table(plot_dict: dict,
                  mass_window: tuple[float, float],
                  ptcone_max: float,
                  etcone_max: float,
                  require_both: bool = True) -> pd.DataFrame:
    steps = [
        ("Baseline", dict()),
        ("Mass window", dict(mass_window=mass_window)),
        ("Iso only", dict(ptcone_max=ptcone_max, etcone_max=etcone_max)),
        ("Mass + Iso", dict(mass_window=mass_window, ptcone_max=ptcone_max, etcone_max=etcone_max)),
    ]
    rows = []
    for name, kw in steps:
        dsel = select_plotdict(plot_dict, require_both=require_both, **kw)
        n_data, n_sig, n_bkg, sb, dmc = summarize(dsel)
        rows.append([name, n_data, n_sig, n_bkg, sb, dmc])
    return pd.DataFrame(rows, columns=["Step", "Data", "Signal", "Background", "S/B", "Data/MC"])


# ============================================================
# 5) Plot helpers (saving + grouping)
# ============================================================

def save_fig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def compose_image_grid(image_paths: list[Path], out_path: Path,
                       nrows: int, ncols: int, titles: list[str] | None = None,
                       figsize: tuple[float, float] = (14, 10)) -> None:
    """
    Create a single multi-panel figure from existing PNGs.
    This satisfies the "use ax" requirement without relying on backend internals.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        if i >= len(image_paths):
            ax.axis("off")
            continue
        img = plt.imread(image_paths[i])
        ax.imshow(img)
        ax.axis("off")
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=12)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 6) Isolation scan
# ============================================================

def scan_isolation(plot_OS: dict, plot_SS: dict,
                   mass_window: tuple[float, float],
                   require_both: bool,
                   ptcone_range: tuple[float, float], ptcone_step: float,
                   etcone_range: tuple[float, float], etcone_step: float,
                   os_sig_eff_min: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Scan (ptcone_max, etcone_max) inside mass_window and choose best point:
      - keep OS signal efficiency >= os_sig_eff_min (relative to mass-only)
      - maximise SS data rejection
      - tie-breaker: smaller (ptcone_max + etcone_max)
    Returns: (df_scan, df_allowed, best_row)
    """

    os_ref = select_plotdict(plot_OS, mass_window=mass_window, require_both=require_both)
    ss_ref = select_plotdict(plot_SS, mass_window=mass_window, require_both=require_both)

    _, os_sig_ref, _, _, _ = summarize(os_ref)
    ss_data_ref, _, _, _, _ = summarize(ss_ref)

    pt_vals = np.arange(ptcone_range[0], ptcone_range[1] + 1e-12, ptcone_step)
    et_vals = np.arange(etcone_range[0], etcone_range[1] + 1e-12, etcone_step)

    rows = []
    for ptc in pt_vals:
        for etc in et_vals:
            os_sel = select_plotdict(plot_OS, mass_window=mass_window,
                                     ptcone_max=float(ptc), etcone_max=float(etc),
                                     require_both=require_both)
            ss_sel = select_plotdict(plot_SS, mass_window=mass_window,
                                     ptcone_max=float(ptc), etcone_max=float(etc),
                                     require_both=require_both)

            os_data, os_sig, os_bkg, os_sb, os_dmc = summarize(os_sel)
            ss_data, ss_sig, ss_bkg, ss_sb, ss_dmc = summarize(ss_sel)

            os_sig_eff = (os_sig / os_sig_ref) if os_sig_ref > 0 else float("nan")
            ss_rej = 1.0 - (ss_data / ss_data_ref) if ss_data_ref > 0 else float("nan")

            rows.append([float(ptc), float(etc), os_sig_eff, ss_rej, os_dmc, os_sb, ss_data])

    df_scan = pd.DataFrame(
        rows, columns=["ptcone_max", "etcone_max", "OS_sig_eff", "SS_rejection", "OS_Data/MC", "OS_S/B", "SS_Data"]
    )

    df_allowed = df_scan[df_scan["OS_sig_eff"] >= os_sig_eff_min].copy()
    df_allowed["tightness"] = df_allowed["ptcone_max"] + df_allowed["etcone_max"]
    df_allowed = df_allowed.sort_values(["SS_rejection", "tightness"], ascending=[False, True])

    if len(df_allowed) == 0:
        raise RuntimeError(
            f"No scan point satisfies OS_sig_eff_min={os_sig_eff_min}. "
            f"Try loosening ISO_SCAN ranges or lowering OS_SIG_EFF_MIN."
        )

    best = df_allowed.iloc[0]
    return df_scan, df_allowed, best


def save_scan_heatmap(df_scan: pd.DataFrame, out_path: Path, value_col: str,
                      title: str, best_ptc: float | None = None, best_etc: float | None = None) -> None:
    """
    Make a simple 2D heatmap over (ptcone, etcone).
    """
    # pivot to grid
    pivot = df_scan.pivot(index="etcone_max", columns="ptcone_max", values=value_col).sort_index(axis=0).sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, origin="lower", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("ptcone_max [GeV]")
    ax.set_ylabel("etcone_max [GeV]")

    # ticks (sparse)
    xticks = np.linspace(0, pivot.shape[1]-1, min(6, pivot.shape[1])).astype(int)
    yticks = np.linspace(0, pivot.shape[0]-1, min(6, pivot.shape[0])).astype(int)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f"{pivot.columns[i]:.2f}" for i in xticks], rotation=45, ha="right")
    ax.set_yticklabels([f"{pivot.index[i]:.2f}" for i in yticks])

    fig.colorbar(im, ax=ax, label=value_col)

    # mark best point (approximate)
    if best_ptc is not None and best_etc is not None:
        try:
            x = list(pivot.columns).index(best_ptc)
            y = list(pivot.index).index(best_etc)
            ax.scatter([x], [y], marker="x", s=100)
        except ValueError:
            pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 7) Cross section (cut-and-count)
# ============================================================

LUMI_FB = 30.6          # [fb^-1]  (lab script)
LUMI_REL_UNC = 0.017    # 1.7%     (lab script)
LUMI_PB = LUMI_FB * 1000.0  # convert to pb^-1

def produced_sumw(produced_event_count_fn, sample_key: str, lumi_fb: float) -> float:
    """
    produced_event_count() prints a number, so we capture stdout and parse it.
    """
    import io
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        produced_event_count_fn(sample_key, lumi_fb)
    txt = buf.getvalue().strip()
    m = re.search(r"([0-9]+)\s*$", txt)
    if not m:
        raise RuntimeError(f"Could not parse produced_event_count output:\n{txt}")
    return float(m.group(1))


def compute_sigma(plot_OS: dict,
                  cfg: dict,
                  produced_event_count_fn,
                  mass_window: tuple[float, float],
                  ptcone_max: float,
                  etcone_max: float,
                  require_both: bool) -> dict:
    """
    Return yields, epsilon, sigma and statistical uncertainty.

    Background definition:
      everything except Data and the PRIMARY signal (Zee or Zmumu).
    """
    primary_signal = cfg["signals"][0]           # "Zmumu" or "Zee"
    primary_key = f"Signal {primary_signal}"

    sel = select_plotdict(plot_OS,
                          mass_window=mass_window,
                          ptcone_max=ptcone_max,
                          etcone_max=etcone_max,
                          require_both=require_both)

    N_selected = float(len(sel["Data"]))         # data is unweighted counts

    # background estimate from MC (weighted)
    N_bkg = 0.0
    Var_bkg = 0.0
    for k, v in sel.items():
        if k == "Data":
            continue
        if k == primary_key:
            continue
        N_bkg += yield_mc(v)
        Var_bkg += yield_mc_var(v)

    N_sig_data = N_selected - N_bkg

    # efficiency from primary signal MC
    W_sig_pass = yield_mc(sel.get(primary_key))
    W_sig_total = produced_sumw(produced_event_count_fn, primary_signal, LUMI_FB)
    epsilon = W_sig_pass / W_sig_total

    sigma_pb = N_sig_data / (epsilon * LUMI_PB)

    # stat: data Poisson + MC bkg stat
    dN_data = math.sqrt(N_selected)
    dN_bkg = math.sqrt(Var_bkg)
    dN_sig = math.sqrt(dN_data**2 + dN_bkg**2)
    dsigma_stat_pb = dN_sig / (epsilon * LUMI_PB)

    return {
        "primary_signal": primary_signal,
        "N_selected": N_selected,
        "N_bkg": N_bkg,
        "N_sig_data": N_sig_data,
        "epsilon": epsilon,
        "sigma_pb": sigma_pb,
        "dsigma_stat_pb": dsigma_stat_pb,
    }


def estimate_syst(plot_OS: dict, cfg: dict, produced_event_count_fn,
                  mass_window: tuple[float, float], ptc: float, etc: float, require_both: bool,
                  nominal_sigma_pb: float) -> tuple[float, list[tuple[str, float]]]:
    """
    "Catch-all" systematic estimate by varying cuts (lab script suggestion).
    Returns (syst_pb, [(label, sigma_pb), ...]).
    """
    variations = [
        ("ptcone -0.5", mass_window, max(ptc - 0.5, 0.0), etc, require_both),
        ("ptcone +0.5", mass_window, ptc + 0.5,           etc, require_both),
        ("etcone -0.5", mass_window, ptc, max(etc - 0.5, 0.0), require_both),
        ("etcone +0.5", mass_window, ptc, etc + 0.5,           require_both),
        ("mass tighter", (mass_window[0] + 2.0, mass_window[1] - 2.0), ptc, etc, require_both),
        ("mass looser",  (mass_window[0] - 2.0, mass_window[1] + 2.0), ptc, etc, require_both),
        ("iso leading only", mass_window, ptc, etc, False),
    ]

    sigmas = []
    for label, mw, v_ptc, v_etc, both in variations:
        out = compute_sigma(plot_OS, cfg, produced_event_count_fn, mw, v_ptc, v_etc, both)
        sigmas.append((label, out["sigma_pb"]))

    syst = max(abs(s - nominal_sigma_pb) for _, s in sigmas) if sigmas else 0.0
    return syst, sigmas


# ============================================================
# 8) Main pipeline per channel
# ============================================================

import math  # after functions for clarity

def run_channel(lepton: str, backend: dict, outdir: Path) -> None:
    cfg = CHANNELS[lepton]
    outdir.mkdir(parents=True, exist_ok=True)

    validate_read_variables = backend["validate_read_variables"]
    analysis_parquet = backend["analysis_parquet"]
    plot_stacked_hist = backend["plot_stacked_hist"]
    produced_event_count = backend["produced_event_count"]

    # ---- validate variables to read ----
    read_vars = validate_read_variables(cfg["string_codes"], BASE_VARS + ISO_VARS)

    # ---- load OS & SS ----
    cut_OS = make_cut_function(lepton=lepton, sign="OS", pt_min=SETTINGS["PT_MIN"])
    cut_SS = make_cut_function(lepton=lepton, sign="SS", pt_min=SETTINGS["PT_MIN"])

    data_OS = analysis_parquet(read_vars, cfg["string_codes"], fraction=SETTINGS["FRACTION"], cut_function=cut_OS)
    data_SS = analysis_parquet(read_vars, cfg["string_codes"], fraction=SETTINGS["FRACTION"], cut_function=cut_SS)

    suffix = infer_suffix(data_OS, "2to4lep")
    print(f"[{lepton}] inferred suffix = {suffix}")

    plot_OS = build_plot_dict(data_OS, suffix=suffix, lepton=lepton)
    plot_SS = build_plot_dict(data_SS, suffix=suffix, lepton=lepton)

    # stable colors: Data (k), Signal1 (b), Signal2 (y), Backgrounds (g,r,m)
    color_list = ["k", "b", "y", "g", "r", "m"]

    # ---- BEFORE plots: leading pT + mass (OS & SS) ----
    before_dir = outdir / "plots_before"
    before_dir.mkdir(exist_ok=True, parents=True)

    pconf = SETTINGS["PLOTS"]["LEADING_PT"]
    fig1, _ = plot_stacked_hist(plot_OS, "lep_pt[0]", color_list,
                                pconf["bins"], pconf["xmin"], pconf["xmax"],
                                f"{cfg['leading_label']} ({lepton} OS, before cuts)",
                                logy=pconf["logy"], show_text=True, residual_plot=True, save_fig=False)
    p1 = before_dir / f"{lepton}_OS_leadingPt_before.png"
    if SETTINGS["SAVE_PLOTS"]:
        save_fig(fig1, p1)

    fig2, _ = plot_stacked_hist(plot_SS, "lep_pt[0]", color_list,
                                pconf["bins"], pconf["xmin"], pconf["xmax"],
                                f"{cfg['leading_label']} ({lepton} SS, before cuts)",
                                logy=pconf["logy"], show_text=True, residual_plot=True, save_fig=False)
    p2 = before_dir / f"{lepton}_SS_leadingPt_before.png"
    if SETTINGS["SAVE_PLOTS"]:
        save_fig(fig2, p2)

    mconf = SETTINGS["PLOTS"]["MASS_FULL"]
    fig3, _ = plot_stacked_hist(plot_OS, "mass", color_list,
                                mconf["bins"], mconf["xmin"], mconf["xmax"],
                                f"mass [GeV] ({lepton} OS, before cuts)",
                                logy=mconf["logy"], show_text=True, residual_plot=True, save_fig=False)
    p3 = before_dir / f"{lepton}_OS_mass_before.png"
    if SETTINGS["SAVE_PLOTS"]:
        save_fig(fig3, p3)

    fig4, _ = plot_stacked_hist(plot_SS, "mass", color_list,
                                mconf["bins"], mconf["xmin"], mconf["xmax"],
                                f"mass [GeV] ({lepton} SS, before cuts)",
                                logy=mconf["logy"], show_text=True, residual_plot=True, save_fig=False)
    p4 = before_dir / f"{lepton}_SS_mass_before.png"
    if SETTINGS["SAVE_PLOTS"]:
        save_fig(fig4, p4)

    if SETTINGS["MAKE_GROUP_FIGURES"] and SETTINGS["SAVE_PLOTS"]:
        compose_image_grid(
            [p1, p2, p3, p4],
            outdir / f"{lepton}_beforeplots_grid.png",
            nrows=2, ncols=2,
            titles=["OS leading pT", "SS leading pT", "OS mass", "SS mass"],
            figsize=(16, 12)
        )

    # ---- ISO scan (or fixed iso) ----
    iso_dir = outdir / "iso_scan"
    iso_dir.mkdir(exist_ok=True, parents=True)

    mass_window = SETTINGS["MASS_WINDOW"]
    require_both = SETTINGS["REQUIRE_BOTH_ISO"]

    if SETTINGS["USE_SCAN"]:
        scan_cfg = SETTINGS["ISO_SCAN"]
        df_scan, df_allowed, best = scan_isolation(
            plot_OS, plot_SS,
            mass_window=mass_window,
            require_both=require_both,
            ptcone_range=scan_cfg["PTCONE_RANGE"], ptcone_step=scan_cfg["PTCONE_STEP"],
            etcone_range=scan_cfg["ETCONE_RANGE"], etcone_step=scan_cfg["ETCONE_STEP"],
            os_sig_eff_min=scan_cfg["OS_SIG_EFF_MIN"],
        )
        best_ptc = float(best["ptcone_max"])
        best_etc = float(best["etcone_max"])
        print(f"[{lepton}] best iso: ptcone<{best_ptc:.3f}, etcone<{best_etc:.3f}  (OS_sig_eff={best['OS_sig_eff']:.4f}, SS_rej={best['SS_rejection']:.4f})")

        if SETTINGS["SAVE_SCAN_TABLES"]:
            df_scan.to_csv(iso_dir / f"{lepton}_iso_scan_full.csv", index=False)
            df_allowed.to_csv(iso_dir / f"{lepton}_iso_scan_allowed.csv", index=False)
            best.to_frame("best").to_csv(iso_dir / f"{lepton}_iso_scan_best.csv")

        # heatmaps
        if SETTINGS["SAVE_PLOTS"]:
            save_scan_heatmap(df_scan, iso_dir / f"{lepton}_heatmap_SS_rejection.png",
                              value_col="SS_rejection",
                              title=f"{lepton}: SS rejection (higher is better)",
                              best_ptc=best_ptc, best_etc=best_etc)
            save_scan_heatmap(df_scan, iso_dir / f"{lepton}_heatmap_OS_sig_eff.png",
                              value_col="OS_sig_eff",
                              title=f"{lepton}: OS signal efficiency",
                              best_ptc=best_ptc, best_etc=best_etc)
    else:
        best_ptc = float(SETTINGS["FIXED_ISO"]["ptcone_max"])
        best_etc = float(SETTINGS["FIXED_ISO"]["etcone_max"])
        print(f"[{lepton}] using FIXED iso: ptcone<{best_ptc:.3f}, etcone<{best_etc:.3f}")

    # ---- Apply best (mass + iso) selection ----
    plot_OS_after = select_plotdict(plot_OS, mass_window=mass_window, ptcone_max=best_ptc, etcone_max=best_etc, require_both=require_both)
    plot_SS_after = select_plotdict(plot_SS, mass_window=mass_window, ptcone_max=best_ptc, etcone_max=best_etc, require_both=require_both)

    # ---- AFTER plots: mass zoom for OS & SS ----
    after_dir = outdir / "plots_after"
    after_dir.mkdir(exist_ok=True, parents=True)

    zconf = SETTINGS["PLOTS"]["MASS_ZOOM"]
    fig5, _ = plot_stacked_hist(plot_OS_after, "mass", color_list,
                                zconf["bins"], zconf["xmin"], zconf["xmax"],
                                f"mass [GeV] ({lepton} OS, after cuts: {mass_window[0]}<m<{mass_window[1]}, ptcone<{best_ptc:.3f}, etcone<{best_etc:.3f})",
                                logy=zconf["logy"], show_text=True, residual_plot=True, save_fig=False)
    p5 = after_dir / f"{lepton}_OS_mass_after.png"
    if SETTINGS["SAVE_PLOTS"]:
        save_fig(fig5, p5)

    fig6, _ = plot_stacked_hist(plot_SS_after, "mass", color_list,
                                zconf["bins"], zconf["xmin"], zconf["xmax"],
                                f"mass [GeV] ({lepton} SS, after cuts: {mass_window[0]}<m<{mass_window[1]}, ptcone<{best_ptc:.3f}, etcone<{best_etc:.3f})",
                                logy=zconf["logy"], show_text=True, residual_plot=True, save_fig=False)
    p6 = after_dir / f"{lepton}_SS_mass_after.png"
    if SETTINGS["SAVE_PLOTS"]:
        save_fig(fig6, p6)

    if SETTINGS["MAKE_GROUP_FIGURES"] and SETTINGS["SAVE_PLOTS"]:
        compose_image_grid(
            [p5, p6],
            outdir / f"{lepton}_aftermass_grid.png",
            nrows=1, ncols=2,
            titles=["OS mass (after cuts)", "SS mass (after cuts)"],
            figsize=(16, 6)
        )

    # ---- Cutflow tables ----
    df_os = cutflow_table(plot_OS, mass_window, best_ptc, best_etc, require_both=require_both)
    df_ss = cutflow_table(plot_SS, mass_window, best_ptc, best_etc, require_both=require_both)

    if SETTINGS["SAVE_TABLES"]:
        df_os.to_csv(outdir / f"{lepton}_cutflow_OS.csv", index=False)
        df_ss.to_csv(outdir / f"{lepton}_cutflow_SS.csv", index=False)

    print(f"\n[{lepton}] === OS cutflow ===")
    print(df_os.to_string(index=False))
    print(f"\n[{lepton}] === SS cutflow ===")
    print(df_ss.to_string(index=False))

    # ---- Cross section ----
    nom = compute_sigma(plot_OS, cfg, produced_event_count,
                        mass_window=mass_window, ptcone_max=best_ptc, etcone_max=best_etc,
                        require_both=require_both)

    sigma_pb = nom["sigma_pb"]
    ds_stat_pb = nom["dsigma_stat_pb"]
    ds_lumi_pb = LUMI_REL_UNC * sigma_pb

    syst_pb, syst_details = estimate_syst(plot_OS, cfg, produced_event_count,
                                          mass_window, best_ptc, best_etc, require_both,
                                          nominal_sigma_pb=sigma_pb)

    result = {
        "lepton": lepton,
        "mass_window": mass_window,
        "require_both_iso": require_both,
        "best_iso": {"ptcone_max": best_ptc, "etcone_max": best_etc},
        "nominal": nom,
        "uncertainties_pb": {"stat": ds_stat_pb, "syst": syst_pb, "lumi": ds_lumi_pb},
        "uncertainties_nb": {"stat": ds_stat_pb/1000.0, "syst": syst_pb/1000.0, "lumi": ds_lumi_pb/1000.0},
        "sigma_pb": sigma_pb,
        "sigma_nb": sigma_pb/1000.0,
        "syst_variations": [{"label": lab, "sigma_pb": s} for lab, s in syst_details],
    }

    print(f"\n[{lepton}] === Cross section (cut-and-count) ===")
    print(f"sigma = {sigma_pb:.3f} ± {ds_stat_pb:.3f} (stat.) ± {syst_pb:.3f} (syst.) ± {ds_lumi_pb:.3f} (lumi.) pb")
    print(f"      = {sigma_pb/1000.0:.4f} ± {ds_stat_pb/1000.0:.4f} ± {syst_pb/1000.0:.4f} ± {ds_lumi_pb/1000.0:.4f} nb")

    if SETTINGS["SAVE_JSON"]:
        import json
        with open(outdir / f"{lepton}_cross_section.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    # Save a short text summary too
    summary_txt = outdir / f"{lepton}_summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"Channel: {lepton}\n")
        f.write(f"Fraction: {SETTINGS['FRACTION']}\n")
        f.write(f"Mass window: {mass_window}\n")
        f.write(f"Isolation: ptcone<{best_ptc:.3f}, etcone<{best_etc:.3f}, require_both={require_both}\n\n")
        f.write("OS cutflow:\n")
        f.write(df_os.to_string(index=False))
        f.write("\n\nSS cutflow:\n")
        f.write(df_ss.to_string(index=False))
        f.write("\n\nCross section:\n")
        f.write(f"sigma = {sigma_pb:.3f} ± {ds_stat_pb:.3f} (stat.) ± {syst_pb:.3f} (syst.) ± {ds_lumi_pb:.3f} (lumi.) pb\n")
        f.write(f"      = {sigma_pb/1000.0:.4f} ± {ds_stat_pb/1000.0:.4f} ± {syst_pb/1000.0:.4f} ± {ds_lumi_pb/1000.0:.4f} nb\n")


# ============================================================
# 9) Entry point
# ============================================================

def main() -> None:
    _ensure_environment()
    backend = _import_backend()

    base_out = Path(SETTINGS["OUTPUT_DIR"]).resolve()
    run_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # one folder per run
    run_out = base_out / f"run_{run_tag}"
    run_out.mkdir(parents=True, exist_ok=True)

    # save settings snapshot
    import json
    with open(run_out / "settings.json", "w", encoding="utf-8") as f:
        json.dump(SETTINGS, f, indent=2)

    for lep in SETTINGS["LEPTONS"]:
        if lep not in CHANNELS:
            raise ValueError(f"Unknown lepton channel: {lep}. Choose from {list(CHANNELS)}.")
        run_channel(lep, backend, run_out / lep)

    print(f"\nAll done. Outputs saved under: {run_out}")


if __name__ == "__main__":
    main()
