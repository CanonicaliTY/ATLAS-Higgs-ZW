#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# v2.0 update: refactored final version with notebook-style tight parquet workflow,
# optional Medium ID consideration, and optional separate data-driven b-bbar-like 
# background study from data control regions.

"""
ATLAS Open Data (Manchester 3rd year lab): Z -> ll dilepton analysis runner
Refactored final version with:
  - notebook-style tight parquet workflow (preselect/write, then read back)
  - optional Medium ID consideration
  - optional, separate data-driven b-bbar-like background study from data control regions

Main workflow (per chosen lepton channel):
  1) Build / reuse a channel-specific "main tight parquet" with baseline preselection.
  2) Read baseline-tight parquet back, split into OS / SS at read stage.
  3) Make BEFORE plots (leading pT, m_ll) for OS & SS.
  4) Scan isolation cuts (or use fixed cuts).
  5) Make AFTER plots, cutflow, and cut-and-count cross section.
  6) Optionally build / reuse a separate data-only control tight parquet and evaluate
     three independent data-driven control-region estimates:
       - wrong flavour (eμ, OS)
       - wrong charge (same flavour as target channel, SS)
       - both wrong (eμ, SS)
     plus their average. This module is kept separate from the main workflow by default.

This design follows the lab notebook's "Write the Data to Disk" methodology:
write a tighter preselection to parquet, then read it back with a smaller set of variables and
apply further cuts. See also the lab script section 7.5 about defining tighter datasets to enable
larger fractions / full datasets.  # see response text for citation
"""

from __future__ import annotations

# ============================================================
# 0) SETTINGS (edit here only)
# ============================================================

SETTINGS = {
    # Which channel(s) to run?
    # ("mu",) ; ("e",) ; ("mu","e")
    "LEPTONS": ("mu",),

    # Fractions
    "FRACTION": 1.0,   # fraction used when READING for the analysis stage
    "PT_MIN": 10.0,    # baseline lepton pT threshold (GeV)

    # Final signal-region selection
    "MASS_WINDOW": (66.0, 116.0),  # GeV
    "REQUIRE_BOTH_ISO": True,

    # Isolation scan
    "USE_SCAN": True,
    "ISO_SCAN": {
        "PTCONE_RANGE": (0.0, 10.0),
        "PTCONE_STEP": 0.25,
        "ETCONE_RANGE": (0.0, 20.0),
        "ETCONE_STEP": 0.25,
        # can be a float or a dict {"mu": ..., "e": ...}
        "OS_SIG_EFF_MIN": {"mu": 0.995, "e": 0.98},
    },
    # used only if USE_SCAN=False
    "FIXED_ISO": {"ptcone_max": 4.5, "etcone_max": 9.25},

    # Optional Medium ID consideration
    "MEDIUM_ID": {
        "APPLY": False,
        # "mc_only_if_available"   : apply to MC only, if field exists
        # "if_available_all"       : apply to data/MC whenever field exists
        # "disabled"               : never apply
        "SCOPE": "mc_only_if_available",
        "FIELD_CANDIDATES": ["lep_isMediumID"],
    },

    # Tight parquet workflow
    "USE_TIGHT_PARQUET": True,
    "TIGHT_PARQUET": {
        # full raw fraction used when BUILDING tight parquet
        "BUILD_FRACTION": 1.0,

        # rebuild even if a complete tight parquet already exists
        "FORCE_REBUILD": False,

        # where tight parquet roots live
        "ROOT_DIR": "../../tight-parquet",

        # optional manual tag override; None -> auto-generated
        "MAIN_TAG": None,
        "CONTROL_TAG": None,

        # what stage to write for the main channel-specific tight parquet
        # "baseline" recommended: same-flavour + trigger + pT + derived mass/qprod
        "MAIN_STAGE": "baseline",

        # keep fields in the MAIN tight parquet (other build-time raw vars are temporary only)
        "MAIN_KEEP_FIELDS": [
            "lep_pt",
            "lep_ptvarcone30",
            "lep_topoetcone20",
            "mass",
            "charge_product",
        ],

        # keep fields in the DATA-ONLY control tight parquet for the optional b-bbar study
        "CONTROL_KEEP_FIELDS": [
            "lep_pt",
            "lep_type",
            "lep_charge",
            "lep_ptvarcone30",
            "lep_topoetcone20",
            "mass",
            "charge_product",
        ],

        # write metadata json/csv
        "WRITE_METADATA": True,
    },

    # Purely additional, separate data-driven control-region study
    "ADDITIONAL_BBAR": {
        "ENABLED": False,  # kept fully separate from the main result unless you turn it on
        "CONTROL_TRIGGER_MODE": "or",  # "or" / "mu" / "e"

        # transfer factors are intentionally exposed because the teacher's guidance here is exploratory
        # default = unity transfer, i.e. count-level comparison / subtraction study
        "TRANSFER_FACTORS": {
            "wrong_flavour": 1.0,
            "wrong_charge": 1.0,
            "both_wrong": 1.0,
        },

        # if not None, also compute a sigma variant with this additional estimate applied:
        # None / "wrong_flavour" / "wrong_charge" / "both_wrong" / "avg3"
        "APPLY_METHOD": None,

        # combine with the main MC background as an *additional* poorly-modelled contribution
        "ADD_TO_MC_BACKGROUND": True,

        # optional outputs
        "SAVE_TABLES": True,
        "SAVE_PLOTS": True,
    },

    # Plot config
    "PLOTS": {
        "LEADING_PT": {"xmin": 0, "xmax": 200, "bins": 50, "logy": True},
        "MASS_FULL": {"xmin": 0, "xmax": 200, "bins": 120, "logy": True},
        "MASS_ZOOM": {"xmin": 60, "xmax": 120, "bins": 60, "logy": True},
    },

    # Outputs
    "OUTPUT_DIR": "output_py",
    "SAVE_PLOTS": True,
    "SAVE_TABLES": True,
    "SAVE_JSON": True,
    "SAVE_SCAN_TABLES": True,
    "MAKE_GROUP_FIGURES": True,

    # Environment setup (optional)
    "AUTO_INSTALL": False,
    "RUN_INSTALL_FROM_ENV_YML": False,
}

# ============================================================
# 1) Environment helpers
# ============================================================

from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parent)  # ensure relative paths are stable

import sys
import subprocess
import shutil
import json
import re
import time
import glob
import math
import datetime
import contextlib
import io

def _pip_install(requirement: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", requirement])

def _ensure_environment() -> None:
    if not SETTINGS["AUTO_INSTALL"] and not SETTINGS["RUN_INSTALL_FROM_ENV_YML"]:
        return

    try:
        import atlasopenmagic  # noqa: F401
    except Exception:
        if SETTINGS["AUTO_INSTALL"]:
            _pip_install("atlasopenmagic")
        else:
            raise

    try:
        import pyarrow  # noqa: F401
        if SETTINGS["AUTO_INSTALL"] and getattr(pyarrow, "__version__", "") != "20.0.0":
            _pip_install("pyarrow==20.0.0")
    except Exception:
        if SETTINGS["AUTO_INSTALL"]:
            _pip_install("pyarrow==20.0.0")
        else:
            raise

    if SETTINGS["RUN_INSTALL_FROM_ENV_YML"]:
        from atlasopenmagic import install_from_environment
        env_file = (Path(__file__).resolve().parent / "../backend/environment.yml").resolve()
        install_from_environment(environment_file=str(env_file))

# ============================================================
# 2) Imports
# ============================================================

import matplotlib
matplotlib.use("Agg")  # save-to-disk workflow for .py scripts

import numpy as np
import pandas as pd
import awkward as ak
import vector
import hist  # noqa: F401
from hist import Hist  # noqa: F401
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator  # noqa: F401
import pyarrow.parquet as pq

def _import_backend():
    here = Path(__file__).resolve()
    parent = here.parent.parent
    if str(parent) not in sys.path:
        sys.path.append(str(parent))

    from backend import (
        get_valid_variables,
        validate_read_variables,
        plot_stacked_hist, plot_histograms, histogram_2d, plot_errorbars,  # noqa: F401
        get_histogram, analysis_parquet, produced_event_count,  # noqa: F401
    )
    return {
        "get_valid_variables": get_valid_variables,
        "validate_read_variables": validate_read_variables,
        "plot_stacked_hist": plot_stacked_hist,
        "analysis_parquet": analysis_parquet,
        "produced_event_count": produced_event_count,
    }

# ============================================================
# 3) Constants / configs
# ============================================================

CHANNELS = {
    "mu": {
        "string_codes": ["2to4lep", "Zmumu", "m10_40_Zmumu", "Ztautau", "ttbar", "Wmunu"],
        "type_sum": 26,  # 13 + 13
        "trigger_field": "trigM",
        "signals": ["Zmumu", "m10_40_Zmumu"],
        "bkgs": ["Ztautau", "ttbar", "Wmunu"],
        "leading_label": r"leading muon $p_T$ [GeV]",
        "title_token": "mu",
    },
    "e": {
        "string_codes": ["2to4lep", "Zee", "m10_40_Zee", "Ztautau", "ttbar", "Wenu"],
        "type_sum": 22,  # 11 + 11
        "trigger_field": "trigE",
        "signals": ["Zee", "m10_40_Zee"],
        "bkgs": ["Ztautau", "ttbar", "Wenu"],
        "leading_label": r"leading electron $p_T$ [GeV]",
        "title_token": "e",
    },
}

RAW_BUILD_VARS_COMMON = [
    "lep_n",
    "lep_pt",
    "lep_eta",
    "lep_phi",
    "lep_e",
    "lep_type",
    "lep_charge",
    "trigM",
    "trigE",
    "lep_ptvarcone30",
    "lep_topoetcone20",
]

LUMI_FB = 30.6
LUMI_REL_UNC = 0.017
LUMI_PB = LUMI_FB * 1000.0

# ============================================================
# 4) General helpers
# ============================================================

def fraction_label(frac: float | int) -> str:
    frac = float(frac)
    return str(frac).replace(".", "_")

def label_float(x: float) -> str:
    return f"{x:.1f}".replace(".", "p")

def now_stamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)

def write_json(path: Path, obj) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)

def write_text(path: Path, text: str) -> None:
    ensure_parent(path)
    path.write_text(text, encoding="utf-8")

def is_data_sample(sample_code: str) -> bool:
    return sample_code in {"2to4lep", "GamGam"}

def medium_id_mode() -> str:
    return SETTINGS["MEDIUM_ID"]["SCOPE"] if SETTINGS["MEDIUM_ID"]["APPLY"] else "disabled"

def iso_eff_threshold_for(lepton: str) -> float:
    val = SETTINGS["ISO_SCAN"]["OS_SIG_EFF_MIN"]
    if isinstance(val, dict):
        return float(val[lepton])
    return float(val)

def get_sample_key_by_prefix(data_dict: dict, sample_code: str) -> str | None:
    matches = [k for k in data_dict.keys() if k.startswith(sample_code + "_")]
    if len(matches) == 0:
        return None
    if len(matches) > 1:
        # choose the shortest / simplest match deterministically
        matches = sorted(matches, key=len)
    return matches[0]

def tight_first_parquet_file(subdir: Path) -> Path:
    files = sorted(subdir.glob("*.parquet"))
    if files:
        return files[0]
    files = sorted(subdir.rglob("*.parquet"))
    if files:
        return files[0]
    raise FileNotFoundError(f"No parquet file found under {subdir}")

def tight_fields_of_subdir(subdir: Path) -> list[str]:
    first = tight_first_parquet_file(subdir)
    table = pq.read_table(first)
    return list(table.schema.names)

def available_raw_fields(sample_code: str, backend: dict) -> list[str]:
    return list(backend["get_valid_variables"](sample_code))

def choose_medium_field(sample_code: str, backend: dict) -> str | None:
    fields = set(available_raw_fields(sample_code, backend))
    for cand in SETTINGS["MEDIUM_ID"]["FIELD_CANDIDATES"]:
        if cand in fields:
            return cand
    return None

def should_apply_medium_id(sample_code: str, medium_field: str | None) -> tuple[bool, str]:
    if not SETTINGS["MEDIUM_ID"]["APPLY"]:
        return False, "disabled"
    if medium_field is None:
        return False, "field_not_available"
    scope = SETTINGS["MEDIUM_ID"]["SCOPE"]
    if scope == "disabled":
        return False, "disabled"
    if scope == "mc_only_if_available":
        if is_data_sample(sample_code):
            return False, "data_sample_skipped"
        return True, "mc_field_available"
    if scope == "if_available_all":
        return True, "field_available"
    return False, f"unknown_scope:{scope}"

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
    if events is None:
        return 0.0
    wf = _weight_field(events)
    if wf is None:
        return float(len(events))
    w = events[wf]
    return float(ak.sum(w * w))

def produced_sumw(produced_event_count_fn, sample_key: str, lumi_fb: float) -> float:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        produced_event_count_fn(sample_key, lumi_fb)
    txt = buf.getvalue().strip()
    m = re.search(r"([0-9]+)\s*$", txt)
    if not m:
        raise RuntimeError(f"Could not parse produced_event_count output:\n{txt}")
    return float(m.group(1))

# ============================================================
# 5) Selection / slim helpers
# ============================================================

def add_mass(data: ak.Array) -> ak.Array:
    p4 = vector.zip({
        "pt": data["lep_pt"],
        "eta": data["lep_eta"],
        "phi": data["lep_phi"],
        "E": data["lep_e"],
    })
    data["mass"] = (p4[:, 0] + p4[:, 1]).M
    return data

def add_charge_product(data: ak.Array) -> ak.Array:
    data["charge_product"] = data["lep_charge"][:, 0] * data["lep_charge"][:, 1]
    return data

def slim_keep_fields(data: ak.Array, keep_fields: list[str]) -> ak.Array:
    final_fields = []
    for f in keep_fields:
        if f in data.fields and f not in final_fields:
            final_fields.append(f)
    wf = _weight_field(data)
    if wf is not None and wf not in final_fields:
        final_fields.append(wf)
    return data[final_fields]

def main_build_keep_fields() -> list[str]:
    keep = list(SETTINGS["TIGHT_PARQUET"]["MAIN_KEEP_FIELDS"])
    # ensure downstream-required fields are present
    for f in ["lep_pt", "lep_ptvarcone30", "lep_topoetcone20", "mass", "charge_product"]:
        if f not in keep:
            keep.append(f)
    return keep

def control_build_keep_fields() -> list[str]:
    keep = list(SETTINGS["TIGHT_PARQUET"]["CONTROL_KEEP_FIELDS"])
    for f in ["lep_pt", "lep_type", "lep_charge", "lep_ptvarcone30", "lep_topoetcone20", "mass", "charge_product"]:
        if f not in keep:
            keep.append(f)
    return keep

def baseline_main_preselection(data: ak.Array, lepton: str, pt_min: float,
                               apply_medium_id: bool, medium_field: str | None) -> ak.Array:
    cfg = CHANNELS[lepton]
    data = data[data["lep_n"] == 2]
    data = data[(data["lep_type"][:, 0] + data["lep_type"][:, 1]) == cfg["type_sum"]]
    data = data[(data["lep_pt"][:, 0] > pt_min) & (data["lep_pt"][:, 1] > pt_min)]
    data = data[data[cfg["trigger_field"]]]
    if apply_medium_id and medium_field is not None and medium_field in data.fields:
        data = data[data[medium_field][:, 0] & data[medium_field][:, 1]]
    data = add_mass(data)
    data = add_charge_product(data)
    return data

def baseline_control_preselection(data: ak.Array, pt_min: float, trigger_mode: str) -> ak.Array:
    data = data[data["lep_n"] == 2]

    # keep only e / mu dileptons (ignore tau-like or anything else)
    t0 = data["lep_type"][:, 0]
    t1 = data["lep_type"][:, 1]
    is_em = ((t0 == 11) | (t0 == 13)) & ((t1 == 11) | (t1 == 13))
    data = data[is_em]

    data = data[(data["lep_pt"][:, 0] > pt_min) & (data["lep_pt"][:, 1] > pt_min)]

    trigger_mode = trigger_mode.lower().strip()
    if trigger_mode == "or":
        data = data[data["trigE"] | data["trigM"]]
    elif trigger_mode == "mu":
        data = data[data["trigM"]]
    elif trigger_mode == "e":
        data = data[data["trigE"]]
    else:
        raise ValueError(f"Unknown CONTROL_TRIGGER_MODE: {trigger_mode}")

    data = add_mass(data)
    data = add_charge_product(data)
    return data

def make_main_build_cut(lepton: str,
                        apply_medium_id: bool,
                        medium_field: str | None,
                        stage: str = "baseline"):
    stage = stage.lower().strip()
    keep = main_build_keep_fields()
    pt_min = SETTINGS["PT_MIN"]

    def _cut(data: ak.Array) -> ak.Array:
        d = baseline_main_preselection(data, lepton=lepton, pt_min=pt_min,
                                       apply_medium_id=apply_medium_id, medium_field=medium_field)
        if stage != "baseline":
            raise ValueError(f"Unsupported MAIN_STAGE={stage!r}; use 'baseline'.")
        return slim_keep_fields(d, keep)

    return _cut

def make_control_build_cut():
    keep = control_build_keep_fields()
    pt_min = SETTINGS["PT_MIN"]
    trigger_mode = SETTINGS["ADDITIONAL_BBAR"]["CONTROL_TRIGGER_MODE"]

    def _cut(data: ak.Array) -> ak.Array:
        d = baseline_control_preselection(data, pt_min=pt_min, trigger_mode=trigger_mode)
        return slim_keep_fields(d, keep)

    return _cut

def make_sign_cut(sign: str):
    sign = sign.upper().strip()

    def _cut(data: ak.Array) -> ak.Array:
        if "charge_product" not in data.fields:
            raise KeyError("charge_product not found in tight parquet; cannot split OS/SS")
        if sign == "OS":
            return data[data["charge_product"] < 0]
        if sign == "SS":
            return data[data["charge_product"] > 0]
        raise ValueError("sign must be 'OS' or 'SS'")

    return _cut

def apply_selection(events: ak.Array | None,
                    mass_window: tuple[float, float] | None = None,
                    ptcone_max: float | None = None,
                    etcone_max: float | None = None,
                    require_both: bool = True) -> ak.Array | None:
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

# ============================================================
# 6) Tight parquet methodology (notebook-style write/read)
# ============================================================

def main_tight_tag(lepton: str) -> str:
    manual = SETTINGS["TIGHT_PARQUET"]["MAIN_TAG"]
    if manual:
        return str(manual)
    mid = medium_id_mode()
    return f"{lepton}_main_{SETTINGS['TIGHT_PARQUET']['MAIN_STAGE']}_pt{label_float(SETTINGS['PT_MIN'])}_mid_{mid}"

def control_tight_tag() -> str:
    manual = SETTINGS["TIGHT_PARQUET"]["CONTROL_TAG"]
    if manual:
        return str(manual)
    trig = SETTINGS["ADDITIONAL_BBAR"]["CONTROL_TRIGGER_MODE"]
    return f"control2lep_pt{label_float(SETTINGS['PT_MIN'])}_trig_{trig}"

def tight_root(kind: str, lepton: str | None = None) -> Path:
    base = (Path(__file__).resolve().parent / SETTINGS["TIGHT_PARQUET"]["ROOT_DIR"]).resolve()
    if kind == "main":
        if lepton is None:
            raise ValueError("lepton required for kind='main'")
        return base / main_tight_tag(lepton)
    if kind == "control":
        return base / control_tight_tag()
    raise ValueError(f"Unknown tight kind: {kind}")

def manifest_path(root: Path) -> Path:
    return root / "_manifest.json"

def read_manifest(root: Path) -> dict | None:
    mp = manifest_path(root)
    if not mp.exists():
        return None
    return json.loads(mp.read_text(encoding="utf-8"))

def manifest_complete(root: Path) -> bool:
    m = read_manifest(root)
    return bool(m and m.get("complete"))

def reset_root_for_rebuild(root: Path) -> None:
    safe_rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

def sample_output_subdirs(tmp_root: Path) -> list[Path]:
    return [p for p in tmp_root.iterdir() if p.is_dir() and not p.name.startswith("_")]

def build_one_sample_to_root(sample_code: str,
                             root: Path,
                             read_vars: list[str],
                             cut_function,
                             backend: dict,
                             fraction: float) -> str | None:
    analysis_parquet = backend["analysis_parquet"]

    tmp_root = root / f"_tmp_{sample_code}_{now_stamp()}"
    safe_rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    analysis_parquet(
        read_vars,
        [sample_code],
        fraction=fraction,
        cut_function=cut_function,
        write_parquet=True,
        output_directory=str(tmp_root),
        return_output=False,
    )

    produced = sample_output_subdirs(tmp_root)
    if len(produced) == 0:
        # nothing selected / nothing written
        safe_rmtree(tmp_root)
        return None
    if len(produced) != 1:
        raise RuntimeError(f"Expected exactly one written subdirectory for {sample_code}, found: {[p.name for p in produced]}")

    src = produced[0]
    dst = root / src.name
    if dst.exists():
        safe_rmtree(dst)
    shutil.move(str(src), str(dst))
    safe_rmtree(tmp_root)
    return dst.name

def ensure_main_tight_parquet(lepton: str, backend: dict) -> Path:
    if not SETTINGS["USE_TIGHT_PARQUET"]:
        raise RuntimeError("ensure_main_tight_parquet called while USE_TIGHT_PARQUET=False")

    root = tight_root("main", lepton)
    force = SETTINGS["TIGHT_PARQUET"]["FORCE_REBUILD"]

    if root.exists() and manifest_complete(root) and not force:
        return root

    reset_root_for_rebuild(root)

    cfg = CHANNELS[lepton]
    stage = SETTINGS["TIGHT_PARQUET"]["MAIN_STAGE"]
    build_fraction = SETTINGS["TIGHT_PARQUET"]["BUILD_FRACTION"]

    sample_rows = []
    subdirs = []

    for sample in cfg["string_codes"]:
        raw_fields = set(available_raw_fields(sample, backend))
        medium_field = choose_medium_field(sample, backend)
        apply_mid, reason = should_apply_medium_id(sample, medium_field)

        needed_raw = list(RAW_BUILD_VARS_COMMON)
        if medium_field is not None and medium_field not in needed_raw:
            needed_raw.append(medium_field)

        read_vars = [v for v in needed_raw if v in raw_fields]
        cut_function = make_main_build_cut(
            lepton=lepton,
            apply_medium_id=apply_mid,
            medium_field=medium_field,
            stage=stage,
        )

        print(f"[{lepton}] building main tight parquet for {sample} -> {root}")
        subdir_name = build_one_sample_to_root(
            sample_code=sample,
            root=root,
            read_vars=read_vars,
            cut_function=cut_function,
            backend=backend,
            fraction=build_fraction,
        )

        sample_rows.append({
            "sample": sample,
            "output_subdir": subdir_name,
            "medium_id_field": medium_field,
            "apply_medium_id": apply_mid,
            "apply_medium_id_reason": reason,
            "read_vars": read_vars,
        })
        if subdir_name is not None:
            subdirs.append(subdir_name)

    manifest = {
        "complete": True,
        "kind": "main",
        "channel": lepton,
        "tag": main_tight_tag(lepton),
        "created_at": now_stamp(),
        "build_fraction": build_fraction,
        "main_stage": stage,
        "settings_snapshot": {
            "PT_MIN": SETTINGS["PT_MIN"],
            "MEDIUM_ID": SETTINGS["MEDIUM_ID"],
            "MAIN_KEEP_FIELDS": main_build_keep_fields(),
        },
        "subdirs": subdirs,
        "samples": sample_rows,
    }
    if SETTINGS["TIGHT_PARQUET"]["WRITE_METADATA"]:
        write_json(manifest_path(root), manifest)
        pd.DataFrame(sample_rows).to_csv(root / "_medium_id_usage.csv", index=False)

    return root

def ensure_control_tight_parquet(backend: dict) -> Path:
    root = tight_root("control")
    force = SETTINGS["TIGHT_PARQUET"]["FORCE_REBUILD"]

    if root.exists() and manifest_complete(root) and not force:
        return root

    reset_root_for_rebuild(root)

    sample = "2to4lep"
    raw_fields = set(available_raw_fields(sample, backend))
    needed_raw = [v for v in RAW_BUILD_VARS_COMMON if v in raw_fields]

    cut_function = make_control_build_cut()

    print(f"[control] building data-only control tight parquet -> {root}")
    subdir_name = build_one_sample_to_root(
        sample_code=sample,
        root=root,
        read_vars=needed_raw,
        cut_function=cut_function,
        backend=backend,
        fraction=SETTINGS["TIGHT_PARQUET"]["BUILD_FRACTION"],
    )

    manifest = {
        "complete": True,
        "kind": "control",
        "tag": control_tight_tag(),
        "created_at": now_stamp(),
        "build_fraction": SETTINGS["TIGHT_PARQUET"]["BUILD_FRACTION"],
        "settings_snapshot": {
            "PT_MIN": SETTINGS["PT_MIN"],
            "CONTROL_TRIGGER_MODE": SETTINGS["ADDITIONAL_BBAR"]["CONTROL_TRIGGER_MODE"],
            "CONTROL_KEEP_FIELDS": control_build_keep_fields(),
        },
        "subdirs": [subdir_name] if subdir_name is not None else [],
        "samples": [{
            "sample": sample,
            "output_subdir": subdir_name,
            "read_vars": needed_raw,
        }],
    }
    if SETTINGS["TIGHT_PARQUET"]["WRITE_METADATA"]:
        write_json(manifest_path(root), manifest)

    return root

def required_main_read_fields() -> list[str]:
    return ["lep_pt", "lep_ptvarcone30", "lep_topoetcone20", "mass", "charge_product", "weight", "totalWeight"]

def required_control_read_fields() -> list[str]:
    return ["lep_pt", "lep_type", "lep_charge", "lep_ptvarcone30", "lep_topoetcone20", "mass", "charge_product", "weight", "totalWeight"]

def load_tight_subdirs(root: Path,
                       subdirs: list[str],
                       needed_fields: list[str],
                       backend: dict,
                       fraction: float,
                       cut_function=None) -> dict:
    analysis_parquet = backend["analysis_parquet"]
    out = {}
    for subdir_name in subdirs:
        subdir = root / subdir_name
        fields = set(tight_fields_of_subdir(subdir))
        read_vars = [v for v in needed_fields if v in fields]
        print(f"[tight read] {subdir_name} with vars: {read_vars}")
        d = analysis_parquet(
            read_variables=read_vars,
            string_code_list=None,
            read_directory=str(root),
            subdirectory_names=[subdir_name],
            fraction=fraction,
            cut_function=cut_function,
            write_parquet=False,
            output_directory=None,
            return_output=True,
        )
        out.update(d)
    return out

def load_main_events(lepton: str, sign: str, backend: dict) -> dict:
    sign = sign.upper().strip()
    if SETTINGS["USE_TIGHT_PARQUET"]:
        root = ensure_main_tight_parquet(lepton, backend)
        man = read_manifest(root)
        subdirs = man["subdirs"]
        return load_tight_subdirs(
            root=root,
            subdirs=subdirs,
            needed_fields=required_main_read_fields(),
            backend=backend,
            fraction=SETTINGS["FRACTION"],
            cut_function=make_sign_cut(sign),
        )

    # Fallback: read raw directly (useful for debugging small fractions)
    cfg = CHANNELS[lepton]
    validate_read_variables = backend["validate_read_variables"]
    analysis_parquet = backend["analysis_parquet"]
    raw_read_vars = validate_read_variables(cfg["string_codes"], RAW_BUILD_VARS_COMMON)
    cut_function = make_raw_sign_cut(lepton, sign)
    return analysis_parquet(raw_read_vars, cfg["string_codes"], fraction=SETTINGS["FRACTION"], cut_function=cut_function)

def load_control_data(backend: dict) -> dict:
    if SETTINGS["USE_TIGHT_PARQUET"]:
        root = ensure_control_tight_parquet(backend)
        man = read_manifest(root)
        subdirs = man["subdirs"]
        return load_tight_subdirs(
            root=root,
            subdirs=subdirs,
            needed_fields=required_control_read_fields(),
            backend=backend,
            fraction=SETTINGS["FRACTION"],
            cut_function=None,
        )

    # Raw fallback (data only)
    analysis_parquet = backend["analysis_parquet"]
    needed = [v for v in RAW_BUILD_VARS_COMMON if v in set(available_raw_fields("2to4lep", backend))]
    return analysis_parquet(needed, ["2to4lep"], fraction=SETTINGS["FRACTION"], cut_function=make_control_build_cut())

# ============================================================
# 7) Raw fallback sign cuts (same as old direct workflow)
# ============================================================

def make_raw_sign_cut(lepton: str, sign: str):
    sign = sign.upper().strip()

    def _cut(data: ak.Array) -> ak.Array:
        d = baseline_main_preselection(
            data,
            lepton=lepton,
            pt_min=SETTINGS["PT_MIN"],
            apply_medium_id=False,  # raw fallback is just for debugging; MediumID comparisons should use tight parquet
            medium_field=None,
        )
        if sign == "OS":
            return d[d["charge_product"] < 0]
        if sign == "SS":
            return d[d["charge_product"] > 0]
        raise ValueError("sign must be 'OS' or 'SS'")

    return _cut

# ============================================================
# 8) Plot dict / summaries
# ============================================================

def build_plot_dict(d: dict, lepton: str) -> dict:
    cfg = CHANNELS[lepton]
    out = {}
    data_key = get_sample_key_by_prefix(d, "2to4lep")
    out["Data"] = d.get(data_key) if data_key else None

    for s in cfg["signals"]:
        k = get_sample_key_by_prefix(d, s)
        out[f"Signal {s}"] = d.get(k) if k else None
    for b in cfg["bkgs"]:
        k = get_sample_key_by_prefix(d, b)
        out[f"Background {b}"] = d.get(k) if k else None
    return out

def select_plotdict(plot_dict: dict, **selection_kwargs) -> dict:
    out = {}
    for k, v in plot_dict.items():
        out[k] = apply_selection(v, **selection_kwargs)
    return out

def summarize(plot_dict: dict) -> tuple[float, float, float, float, float]:
    n_data = yield_data(plot_dict.get("Data"))
    n_sig = 0.0
    n_bkg = 0.0
    for k, v in plot_dict.items():
        if k.startswith("Signal"):
            n_sig += yield_mc(v)
        elif k.startswith("Background"):
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
# 9) Plot helpers
# ============================================================

def save_fig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    ensure_parent(path)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def compose_image_grid(image_paths: list[Path], out_path: Path,
                       nrows: int, ncols: int,
                       titles: list[str] | None = None,
                       figsize: tuple[float, float] = (14, 10)) -> None:
    ensure_parent(out_path)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        if i >= len(image_paths) or not image_paths[i].exists():
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
# 10) Isolation scan
# ============================================================

def scan_isolation(plot_OS: dict, plot_SS: dict,
                   mass_window: tuple[float, float],
                   require_both: bool,
                   ptcone_range: tuple[float, float], ptcone_step: float,
                   etcone_range: tuple[float, float], etcone_step: float,
                   os_sig_eff_min: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
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

            _, os_sig, _, os_sb, os_dmc = summarize(os_sel)
            ss_data, _, _, _, _ = summarize(ss_sel)

            os_sig_eff = (os_sig / os_sig_ref) if os_sig_ref > 0 else float("nan")
            ss_rej = 1.0 - (ss_data / ss_data_ref) if ss_data_ref > 0 else float("nan")

            rows.append([float(ptc), float(etc), os_sig_eff, ss_rej, os_dmc, os_sb, ss_data])

    df_scan = pd.DataFrame(
        rows,
        columns=["ptcone_max", "etcone_max", "OS_sig_eff", "SS_rejection", "OS_Data/MC", "OS_S/B", "SS_Data"]
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
    pivot = df_scan.pivot(index="etcone_max", columns="ptcone_max", values=value_col).sort_index(axis=0).sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, origin="lower", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("ptcone_max [GeV]")
    ax.set_ylabel("etcone_max [GeV]")

    xticks = np.linspace(0, pivot.shape[1] - 1, min(6, pivot.shape[1])).astype(int)
    yticks = np.linspace(0, pivot.shape[0] - 1, min(6, pivot.shape[0])).astype(int)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f"{pivot.columns[i]:.2f}" for i in xticks], rotation=45, ha="right")
    ax.set_yticklabels([f"{pivot.index[i]:.2f}" for i in yticks])

    fig.colorbar(im, ax=ax, label=value_col)

    if best_ptc is not None and best_etc is not None:
        try:
            x = list(pivot.columns).index(best_ptc)
            y = list(pivot.index).index(best_etc)
            ax.scatter([x], [y], marker="x", s=100)
        except ValueError:
            pass

    ensure_parent(out_path)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# 11) Cross section
# ============================================================

def compute_sigma(plot_OS: dict,
                  cfg: dict,
                  produced_event_count_fn,
                  mass_window: tuple[float, float],
                  ptcone_max: float,
                  etcone_max: float,
                  require_both: bool,
                  extra_bkg: float = 0.0) -> dict:
    primary_signal = cfg["signals"][0]
    primary_key = f"Signal {primary_signal}"

    sel = select_plotdict(plot_OS,
                          mass_window=mass_window,
                          ptcone_max=ptcone_max,
                          etcone_max=etcone_max,
                          require_both=require_both)

    N_selected = float(len(sel["Data"]))

    N_bkg = 0.0
    Var_bkg = 0.0
    for k, v in sel.items():
        if k == "Data":
            continue
        if k == primary_key:
            continue
        N_bkg += yield_mc(v)
        Var_bkg += yield_mc_var(v)

    N_bkg_total = N_bkg + float(extra_bkg)
    N_sig_data = N_selected - N_bkg_total

    W_sig_pass = yield_mc(sel.get(primary_key))
    W_sig_total = produced_sumw(produced_event_count_fn, primary_signal, LUMI_FB)
    epsilon = W_sig_pass / W_sig_total

    sigma_pb = N_sig_data / (epsilon * LUMI_PB)

    dN_data = math.sqrt(N_selected)
    dN_bkg = math.sqrt(Var_bkg)
    dN_sig = math.sqrt(dN_data ** 2 + dN_bkg ** 2)
    dsigma_stat_pb = dN_sig / (epsilon * LUMI_PB)

    return {
        "primary_signal": primary_signal,
        "N_selected": N_selected,
        "N_bkg_mc": N_bkg,
        "N_bkg_extra": float(extra_bkg),
        "N_bkg_total": N_bkg_total,
        "N_sig_data": N_sig_data,
        "epsilon": epsilon,
        "sigma_pb": sigma_pb,
        "dsigma_stat_pb": dsigma_stat_pb,
    }

def estimate_syst(plot_OS: dict, cfg: dict, produced_event_count_fn,
                  mass_window: tuple[float, float], ptc: float, etc: float, require_both: bool,
                  nominal_sigma_pb: float) -> tuple[float, list[tuple[str, float]]]:
    variations = [
        ("ptcone -0.5", mass_window, max(ptc - 0.5, 0.0), etc, require_both),
        ("ptcone +0.5", mass_window, ptc + 0.5, etc, require_both),
        ("etcone -0.5", mass_window, ptc, max(etc - 0.5, 0.0), require_both),
        ("etcone +0.5", mass_window, ptc, etc + 0.5, require_both),
        ("mass tighter", (mass_window[0] + 2.0, mass_window[1] - 2.0), ptc, etc, require_both),
        ("mass looser", (mass_window[0] - 2.0, mass_window[1] + 2.0), ptc, etc, require_both),
        ("iso leading only", mass_window, ptc, etc, False),
    ]

    sigmas = []
    for label, mw, v_ptc, v_etc, both in variations:
        out = compute_sigma(plot_OS, cfg, produced_event_count_fn, mw, v_ptc, v_etc, both)
        sigmas.append((label, out["sigma_pb"]))

    syst = max(abs(s - nominal_sigma_pb) for _, s in sigmas) if sigmas else 0.0
    return syst, sigmas

# ============================================================
# 12) Additional, separate data-driven b-bbar-like study
# ============================================================

def control_region_masks(events: ak.Array, lepton: str) -> dict[str, ak.Array]:
    """
    Independent control regions relative to the same-flavour OS signal definition:
      wrong_flavour : opposite-flavour, OS   (eμ / μe, opposite sign)
      wrong_charge  : target same-flavour, SS
      both_wrong    : opposite-flavour, SS
    """
    cfg = CHANNELS[lepton]
    t0 = events["lep_type"][:, 0]
    t1 = events["lep_type"][:, 1]
    type_sum = t0 + t1
    opposite_flavour = ((t0 == 11) & (t1 == 13)) | ((t0 == 13) & (t1 == 11))
    target_same_flavour = (type_sum == cfg["type_sum"])
    qprod = events["charge_product"]
    os_mask = (qprod < 0)
    ss_mask = (qprod > 0)

    return {
        "wrong_flavour": events[opposite_flavour & os_mask],
        "wrong_charge": events[target_same_flavour & ss_mask],
        "both_wrong": events[opposite_flavour & ss_mask],
    }

def bbar_additional_study(control_data_events: ak.Array,
                          lepton: str,
                          nominal_sigma: dict,
                          plot_OS: dict,
                          cfg: dict,
                          produced_event_count_fn,
                          best_ptc: float,
                          best_etc: float,
                          outdir: Path) -> dict:
    """
    Purely additional / exploratory.
    It does NOT affect the main result unless SETTINGS["ADDITIONAL_BBAR"]["APPLY_METHOD"] is set.

    Current default interpretation:
      - Count data control-region events after the same mass+iso selection.
      - Convert each into an additional background estimate with an adjustable transfer factor
        (default unity, because the teacher's guidance here is exploratory / can be refined later).
      - Compare the effect of subtracting each estimate separately, and also their average.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    final_sel = apply_selection(
        control_data_events,
        mass_window=SETTINGS["MASS_WINDOW"],
        ptcone_max=best_ptc,
        etcone_max=best_etc,
        require_both=SETTINGS["REQUIRE_BOTH_ISO"],
    )

    regions = control_region_masks(final_sel, lepton)
    tf = SETTINGS["ADDITIONAL_BBAR"]["TRANSFER_FACTORS"]

    rows = []
    raw_counts = {}
    for name, ev in regions.items():
        n = float(len(ev))
        raw_counts[name] = n
        est = n * float(tf[name])
        sigma_alt = compute_sigma(
            plot_OS, cfg, produced_event_count_fn,
            SETTINGS["MASS_WINDOW"], best_ptc, best_etc, SETTINGS["REQUIRE_BOTH_ISO"],
            extra_bkg=(est if SETTINGS["ADDITIONAL_BBAR"]["ADD_TO_MC_BACKGROUND"] else 0.0),
        )
        rows.append({
            "method": name,
            "control_count": n,
            "transfer_factor": float(tf[name]),
            "estimated_additional_bkg": est,
            "sigma_pb_if_applied": sigma_alt["sigma_pb"],
            "sigma_shift_pb": sigma_alt["sigma_pb"] - nominal_sigma["sigma_pb"],
        })

    avg3 = float(np.mean([raw_counts["wrong_flavour"] * tf["wrong_flavour"],
                          raw_counts["wrong_charge"] * tf["wrong_charge"],
                          raw_counts["both_wrong"] * tf["both_wrong"]]))
    sigma_avg3 = compute_sigma(
        plot_OS, cfg, produced_event_count_fn,
        SETTINGS["MASS_WINDOW"], best_ptc, best_etc, SETTINGS["REQUIRE_BOTH_ISO"],
        extra_bkg=(avg3 if SETTINGS["ADDITIONAL_BBAR"]["ADD_TO_MC_BACKGROUND"] else 0.0),
    )
    rows.append({
        "method": "avg3",
        "control_count": np.mean([raw_counts["wrong_flavour"], raw_counts["wrong_charge"], raw_counts["both_wrong"]]),
        "transfer_factor": np.nan,
        "estimated_additional_bkg": avg3,
        "sigma_pb_if_applied": sigma_avg3["sigma_pb"],
        "sigma_shift_pb": sigma_avg3["sigma_pb"] - nominal_sigma["sigma_pb"],
    })

    df = pd.DataFrame(rows)

    if SETTINGS["ADDITIONAL_BBAR"]["SAVE_TABLES"]:
        df.to_csv(outdir / f"{lepton}_additional_bbar_estimates.csv", index=False)

    if SETTINGS["ADDITIONAL_BBAR"]["SAVE_PLOTS"]:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(df["method"], df["estimated_additional_bkg"])
        ax.set_ylabel("estimated additional background events")
        ax.set_title(f"{lepton}: additional data-driven b-bbar-like estimates")
        fig.tight_layout()
        fig.savefig(outdir / f"{lepton}_additional_bbar_estimates.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    apply_method = SETTINGS["ADDITIONAL_BBAR"]["APPLY_METHOD"]
    applied = None
    if apply_method is not None:
        row = df[df["method"] == apply_method]
        if len(row) != 1:
            raise ValueError(f"APPLY_METHOD={apply_method!r} not found in additional bbar table")
        applied = row.iloc[0].to_dict()

    result = {
        "table": df.to_dict(orient="records"),
        "applied_method": apply_method,
        "applied_row": applied,
        "note": (
            "Exploratory additional data-driven study. "
            "Control-region definitions currently use:\n"
            "  wrong_flavour = eμ OS,\n"
            "  wrong_charge = same-flavour SS (target channel),\n"
            "  both_wrong = eμ SS.\n"
            "Transfer factors are user-adjustable and default to unity."
        ),
    }
    if SETTINGS["SAVE_JSON"]:
        write_json(outdir / f"{lepton}_additional_bbar_estimates.json", result)
    return result

# ============================================================
# 13) Main pipeline per channel
# ============================================================

def run_channel(lepton: str, backend: dict, outdir: Path) -> None:
    cfg = CHANNELS[lepton]
    outdir.mkdir(parents=True, exist_ok=True)

    plot_stacked_hist = backend["plot_stacked_hist"]
    produced_event_count = backend["produced_event_count"]

    # ---- load OS / SS from baseline tight parquet (preferred) or raw fallback ----
    data_OS = load_main_events(lepton, "OS", backend)
    data_SS = load_main_events(lepton, "SS", backend)

    # log-like info
    key_data = get_sample_key_by_prefix(data_OS, "2to4lep")
    suffix_info = key_data.split("2to4lep_", 1)[1] if key_data else "unknown"
    print(f"[{lepton}] inferred suffix = {suffix_info}")

    plot_OS = build_plot_dict(data_OS, lepton=lepton)
    plot_SS = build_plot_dict(data_SS, lepton=lepton)

    color_list = ["k", "b", "y", "g", "r", "m"]

    # ---- BEFORE plots ----
    before_dir = outdir / "plots_before"
    before_dir.mkdir(parents=True, exist_ok=True)

    pconf = SETTINGS["PLOTS"]["LEADING_PT"]
    fig1, _ = plot_stacked_hist(
        plot_OS, "lep_pt[0]", color_list,
        pconf["bins"], pconf["xmin"], pconf["xmax"],
        f"{cfg['leading_label']} ({lepton} OS, before cuts)",
        logy=pconf["logy"], show_text=True, residual_plot=True, save_fig=False
    )
    p1 = before_dir / f"{lepton}_OS_leadingPt_before.png"
    if SETTINGS["SAVE_PLOTS"]:
        save_fig(fig1, p1)

    fig2, _ = plot_stacked_hist(
        plot_SS, "lep_pt[0]", color_list,
        pconf["bins"], pconf["xmin"], pconf["xmax"],
        f"{cfg['leading_label']} ({lepton} SS, before cuts)",
        logy=pconf["logy"], show_text=True, residual_plot=True, save_fig=False
    )
    p2 = before_dir / f"{lepton}_SS_leadingPt_before.png"
    if SETTINGS["SAVE_PLOTS"]:
        save_fig(fig2, p2)

    mconf = SETTINGS["PLOTS"]["MASS_FULL"]
    fig3, _ = plot_stacked_hist(
        plot_OS, "mass", color_list,
        mconf["bins"], mconf["xmin"], mconf["xmax"],
        f"mass [GeV] ({lepton} OS, before cuts)",
        logy=mconf["logy"], show_text=True, residual_plot=True, save_fig=False
    )
    p3 = before_dir / f"{lepton}_OS_mass_before.png"
    if SETTINGS["SAVE_PLOTS"]:
        save_fig(fig3, p3)

    fig4, _ = plot_stacked_hist(
        plot_SS, "mass", color_list,
        mconf["bins"], mconf["xmin"], mconf["xmax"],
        f"mass [GeV] ({lepton} SS, before cuts)",
        logy=mconf["logy"], show_text=True, residual_plot=True, save_fig=False
    )
    p4 = before_dir / f"{lepton}_SS_mass_before.png"
    if SETTINGS["SAVE_PLOTS"]:
        save_fig(fig4, p4)

    if SETTINGS["MAKE_GROUP_FIGURES"] and SETTINGS["SAVE_PLOTS"]:
        compose_image_grid(
            [p1, p2, p3, p4],
            outdir / f"{lepton}_beforeplots_grid.png",
            nrows=2, ncols=2,
            titles=["OS leading pT", "SS leading pT", "OS mass", "SS mass"],
            figsize=(16, 12),
        )

    # ---- isolation scan / fixed iso ----
    iso_dir = outdir / "iso_scan"
    iso_dir.mkdir(parents=True, exist_ok=True)

    mass_window = SETTINGS["MASS_WINDOW"]
    require_both = SETTINGS["REQUIRE_BOTH_ISO"]

    if SETTINGS["USE_SCAN"]:
        scan_cfg = SETTINGS["ISO_SCAN"]
        eff_min = iso_eff_threshold_for(lepton)
        df_scan, df_allowed, best = scan_isolation(
            plot_OS=plot_OS,
            plot_SS=plot_SS,
            mass_window=mass_window,
            require_both=require_both,
            ptcone_range=scan_cfg["PTCONE_RANGE"],
            ptcone_step=scan_cfg["PTCONE_STEP"],
            etcone_range=scan_cfg["ETCONE_RANGE"],
            etcone_step=scan_cfg["ETCONE_STEP"],
            os_sig_eff_min=eff_min,
        )
        best_ptc = float(best["ptcone_max"])
        best_etc = float(best["etcone_max"])
        print(f"[{lepton}] best iso: ptcone<{best_ptc:.3f}, etcone<{best_etc:.3f}  (OS_sig_eff={best['OS_sig_eff']:.4f}, SS_rej={best['SS_rejection']:.4f})")

        if SETTINGS["SAVE_SCAN_TABLES"]:
            df_scan.to_csv(iso_dir / f"{lepton}_iso_scan_full.csv", index=False)
            df_allowed.to_csv(iso_dir / f"{lepton}_iso_scan_allowed.csv", index=False)
            best.to_frame("best").to_csv(iso_dir / f"{lepton}_iso_scan_best.csv")

        if SETTINGS["SAVE_PLOTS"]:
            save_scan_heatmap(df_scan, iso_dir / f"{lepton}_heatmap_SS_rejection.png",
                              value_col="SS_rejection",
                              title=f"{lepton}: SS rejection",
                              best_ptc=best_ptc, best_etc=best_etc)
            save_scan_heatmap(df_scan, iso_dir / f"{lepton}_heatmap_OS_sig_eff.png",
                              value_col="OS_sig_eff",
                              title=f"{lepton}: OS signal efficiency",
                              best_ptc=best_ptc, best_etc=best_etc)
    else:
        best_ptc = float(SETTINGS["FIXED_ISO"]["ptcone_max"])
        best_etc = float(SETTINGS["FIXED_ISO"]["etcone_max"])
        df_scan = None
        df_allowed = None
        best = None

    # ---- AFTER plots: mass zoom OS/SS ----
    after_dir = outdir / "plots_after"
    after_dir.mkdir(parents=True, exist_ok=True)

    plot_OS_after = select_plotdict(plot_OS, mass_window=mass_window,
                                    ptcone_max=best_ptc, etcone_max=best_etc,
                                    require_both=require_both)
    plot_SS_after = select_plotdict(plot_SS, mass_window=mass_window,
                                    ptcone_max=best_ptc, etcone_max=best_etc,
                                    require_both=require_both)

    zconf = SETTINGS["PLOTS"]["MASS_ZOOM"]
    fig5, _ = plot_stacked_hist(
        plot_OS_after, "mass", color_list,
        zconf["bins"], zconf["xmin"], zconf["xmax"],
        f"mass [GeV] ({lepton} OS after cuts: {mass_window[0]:.1f}<m<{mass_window[1]:.1f}, ptcone<{best_ptc:.3f}, etcone<{best_etc:.3f})",
        logy=zconf["logy"], show_text=True, residual_plot=True, save_fig=False
    )
    p5 = after_dir / f"{lepton}_OS_mass_after.png"
    if SETTINGS["SAVE_PLOTS"]:
        save_fig(fig5, p5)

    fig6, _ = plot_stacked_hist(
        plot_SS_after, "mass", color_list,
        zconf["bins"], zconf["xmin"], zconf["xmax"],
        f"mass [GeV] ({lepton} SS after cuts: {mass_window[0]:.1f}<m<{mass_window[1]:.1f}, ptcone<{best_ptc:.3f}, etcone<{best_etc:.3f})",
        logy=zconf["logy"], show_text=True, residual_plot=True, save_fig=False
    )
    p6 = after_dir / f"{lepton}_SS_mass_after.png"
    if SETTINGS["SAVE_PLOTS"]:
        save_fig(fig6, p6)

    if SETTINGS["MAKE_GROUP_FIGURES"] and SETTINGS["SAVE_PLOTS"]:
        compose_image_grid(
            [p5, p6],
            outdir / f"{lepton}_aftermass_grid.png",
            nrows=1, ncols=2,
            titles=["OS mass after cuts", "SS mass after cuts"],
            figsize=(14, 6),
        )

    # ---- cutflow ----
    os_cf = cutflow_table(plot_OS, mass_window=mass_window,
                          ptcone_max=best_ptc, etcone_max=best_etc,
                          require_both=require_both)
    ss_cf = cutflow_table(plot_SS, mass_window=mass_window,
                          ptcone_max=best_ptc, etcone_max=best_etc,
                          require_both=require_both)

    print(f"\n[{lepton}] === OS cutflow ===")
    print(os_cf.to_string(index=False))
    print(f"\n[{lepton}] === SS cutflow ===")
    print(ss_cf.to_string(index=False))

    if SETTINGS["SAVE_TABLES"]:
        os_cf.to_csv(outdir / f"{lepton}_cutflow_OS.csv", index=False)
        ss_cf.to_csv(outdir / f"{lepton}_cutflow_SS.csv", index=False)

    # ---- main cross section ----
    sigma_nom = compute_sigma(
        plot_OS=plot_OS,
        cfg=cfg,
        produced_event_count_fn=produced_event_count,
        mass_window=mass_window,
        ptcone_max=best_ptc,
        etcone_max=best_etc,
        require_both=require_both,
        extra_bkg=0.0,
    )
    syst_pb, syst_list = estimate_syst(
        plot_OS=plot_OS,
        cfg=cfg,
        produced_event_count_fn=produced_event_count,
        mass_window=mass_window,
        ptc=best_ptc,
        etc=best_etc,
        require_both=require_both,
        nominal_sigma_pb=sigma_nom["sigma_pb"],
    )
    lumi_pb = LUMI_REL_UNC * sigma_nom["sigma_pb"]

    print(f"\n[{lepton}] === Cross section (cut-and-count) ===")
    print(f"sigma = {sigma_nom['sigma_pb']:.3f} ± {sigma_nom['dsigma_stat_pb']:.3f} (stat.) ± {syst_pb:.3f} (syst.) ± {lumi_pb:.3f} (lumi.) pb")
    print(f"      = {sigma_nom['sigma_pb']/1000.0:.4f} ± {sigma_nom['dsigma_stat_pb']/1000.0:.4f} ± {syst_pb/1000.0:.4f} ± {lumi_pb/1000.0:.4f} nb")

    # ---- optional additional data-driven study (separate) ----
    additional_result = None
    if SETTINGS["ADDITIONAL_BBAR"]["ENABLED"]:
        control_dict = load_control_data(backend)
        data_key = get_sample_key_by_prefix(control_dict, "2to4lep")
        control_events = control_dict[data_key]
        add_dir = outdir / "additional_bbar"
        additional_result = bbar_additional_study(
            control_data_events=control_events,
            lepton=lepton,
            nominal_sigma=sigma_nom,
            plot_OS=plot_OS,
            cfg=cfg,
            produced_event_count_fn=produced_event_count,
            best_ptc=best_ptc,
            best_etc=best_etc,
            outdir=add_dir,
        )
        if additional_result["applied_row"] is not None:
            row = additional_result["applied_row"]
            print(f"\n[{lepton}] === Additional data-driven study applied ({row['method']}) ===")
            print(f"estimated extra bkg = {row['estimated_additional_bkg']:.3f}")
            print(f"sigma shift         = {row['sigma_shift_pb']:.3f} pb")

    # ---- write summary ----
    summary_txt = []
    summary_txt.append(f"[{lepton}] best_ptcone = {best_ptc:.4f}")
    summary_txt.append(f"[{lepton}] best_etcone = {best_etc:.4f}")
    summary_txt.append("")
    summary_txt.append("OS cutflow:")
    summary_txt.append(os_cf.to_string(index=False))
    summary_txt.append("")
    summary_txt.append("SS cutflow:")
    summary_txt.append(ss_cf.to_string(index=False))
    summary_txt.append("")
    summary_txt.append("Nominal cross section:")
    summary_txt.append(
        f"sigma = {sigma_nom['sigma_pb']:.3f} ± {sigma_nom['dsigma_stat_pb']:.3f} (stat.) "
        f"± {syst_pb:.3f} (syst.) ± {lumi_pb:.3f} (lumi.) pb"
    )
    if additional_result is not None:
        summary_txt.append("")
        summary_txt.append("Additional data-driven b-bbar-like study:")
        summary_txt.append(json.dumps(additional_result, indent=2, default=str))

    write_text(outdir / f"{lepton}_summary.txt", "\n".join(summary_txt))

    if SETTINGS["SAVE_JSON"]:
        write_json(outdir / f"{lepton}_cross_section.json", {
            "channel": lepton,
            "best_ptcone": best_ptc,
            "best_etcone": best_etc,
            "mass_window": mass_window,
            "require_both_iso": require_both,
            "nominal": sigma_nom,
            "syst_pb": syst_pb,
            "lumi_pb": lumi_pb,
            "syst_variations": [{"label": k, "sigma_pb": v} for k, v in syst_list],
            "additional_bbar": additional_result,
        })

# ============================================================
# 14) Main entry point
# ============================================================

def main() -> None:
    _ensure_environment()
    backend = _import_backend()

    run_root = (Path(__file__).resolve().parent / SETTINGS["OUTPUT_DIR"] / f"run_{now_stamp()}").resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    # keep settings snapshot
    if SETTINGS["SAVE_JSON"]:
        write_json(run_root / "settings.json", SETTINGS)

    for lep in SETTINGS["LEPTONS"]:
        run_channel(lep, backend, run_root / lep)

    print(f"\nAll done. Outputs saved under: {run_root}")

if __name__ == "__main__":
    main()
