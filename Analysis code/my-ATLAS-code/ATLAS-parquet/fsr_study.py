from __future__ import annotations

"""
Focused additional-research workflow for the muon-channel FSR / bremsstrahlung study.

This script is intentionally separate from the main analysis pipeline so that the core lab
workflow stays stable and this additional study remains readable.

What it does
------------
1. Build (or reuse) a dedicated tight parquet that keeps the 4-vector information needed for
   FSR-style mass recovery.
2. Load OS dimuon events for data + MC.
3. Construct a control sample: events that fail the nominal isolation point but pass a looser one.
4. Build toy FSR-recovery masses by treating topoetcone20 as a collinear photon proxy.
5. Save before/after stacked-mass plots and a small sigma scan around the nominal cut.

Important physics note
----------------------
This is a toy study, not a precision photon reconstruction.  The correction interprets
lep_topoetcone20 as a near-muon radiated-photon proxy and adds it back collinearly to the
muon 4-vector.  It is therefore best viewed as a hypothesis test.
"""

from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import vector

from config import CHANNELS, LUMI_FB, LUMI_PB, SCRIPT_DIR, SETTINGS
from cross_section import compute_sigma_from_selected
from parquet_io import (
    available_raw_fields,
    build_one_sample_to_root,
    choose_medium_field,
    load_tight_subdirs,
    manifest_complete,
    read_manifest,
    reset_root_for_rebuild,
    should_apply_medium_id,
)
from scan import generate_scan_points
from selections import baseline_main_preselection, make_sign_cut, slim_keep_fields
from utils import (
    ensure_environment,
    ensure_script_directory,
    get_sample_key_by_prefix,
    import_backend,
    log_step,
    now_stamp,
    progress_iter,
    write_json,
    write_text,
    yield_data,
    yield_mc,
)
from visualisation import save_fig, save_scan_heatmap


FSR_SETTINGS = {
    # Main physics choice
    "LEPTON": "mu",

    # Fractions
    "BUILD_FRACTION": 1.0,
    "READ_FRACTION": 1.0,

    # Dedicated parquet for the FSR study
    "FORCE_REBUILD": True,
    "ROOT_DIR": "../../tight-parquet-fsr",

    # Baseline dimuon preselection
    "PT_MIN": float(SETTINGS["PT_MIN"]),
    "APPLY_MEDIUM_ID": bool(SETTINGS["MEDIUM_ID"]["APPLY"]),

    # Isolation working points
    "NOMINAL_ISO": {
        "ptcone_max": float(SETTINGS["FIXED_ISO"]["ptcone_max"]),
        "etcone_max": float(SETTINGS["FIXED_ISO"]["etcone_max"]),
    },
    "LOOSE_ISO": {
        # Set this to (15, 15) if you want to reproduce the older notebook/control plot.
        "ptcone_max": 15.0,
        "etcone_max": 15.0,
    },
    "REQUIRE_BOTH_ISO": True,

    # Signal-region mass window used for the sigma study
    "MASS_WINDOW": tuple(SETTINGS["MASS_WINDOW"]),

    # Toy FSR proxy model
    "FSR": {
        # Only apply the correction to events below this *raw* mass.
        "APPLY_BELOW_MASS": 80.0,
        # Scale factor for topoetcone20 -> recovered photon pT.
        "ETCONE_SCALE": 1.0,
        # Compare two toy modes:
        #   maxcone: add to the muon with the larger topoetcone20 only
        #   both:    add to both muons
        "MODES": ("maxcone", "both"),
    },

    # Sigma scan around the nominal point (MC-only background subtraction; no extra DD term).
    "SCAN": {
        "RUN": True,
        "MODE": "local_box",
        "PTCONE_RANGE": (0.0, 10.0),
        "PTCONE_STEP": 1.0,
        "ETCONE_RANGE": (0.0, 20.0),
        "ETCONE_STEP": 1.0,
        "LOCAL_BOX_PTCONE_HALF_WIDTH": 2.0,
        "LOCAL_BOX_ETCONE_HALF_WIDTH": 2.0,
    },

    # Plots / outputs
    "OUTPUT_DIR": "output_fsr",
    "SAVE_PLOTS": True,
    "MASS_PLOT": {
        "FULL": {"xmin": 0.0, "xmax": 200.0, "bins": 120, "logy": True},
        "ZOOM": {"xmin": 40.0, "xmax": 120.0, "bins": 80, "logy": True},
    },
    "WINDOWS": [(10.0, 40.0), (40.0, 66.0), (66.0, 80.0), (80.0, 100.0), (100.0, 116.0)],
}


def _fsr_keep_fields() -> list[str]:
    keep = [
        "lep_pt",
        "lep_eta",
        "lep_phi",
        "lep_e",
        "lep_type",
        "lep_charge",
        "lep_ptvarcone30",
        "lep_topoetcone20",
        "mass",
        "charge_product",
    ]
    return keep


def _fsr_root(lepton: str) -> Path:
    base = (SCRIPT_DIR / FSR_SETTINGS["ROOT_DIR"]).resolve()
    tag = (
        f"{lepton}_fsrstudy"
        f"_pt{FSR_SETTINGS['PT_MIN']:.1f}"
        f"_mid_{'on' if FSR_SETTINGS['APPLY_MEDIUM_ID'] else 'off'}"
    ).replace(".", "p")
    return base / tag


def _fsr_manifest_path(root: Path) -> Path:
    return root / "_manifest.json"


def _fsr_build_cut(
    lepton: str,
    apply_medium_id: bool,
    medium_field: str | None,
    required_input_fields: list[str] | None = None,
):
    # Keep the final physics fields *and* the raw fields needed by the build-time
    # preselection.  The backend parquet builder may re-apply the cut function on
    # already-written chunks; if lep_n / trig / Medium-ID fields are dropped too
    # early, baseline_main_preselection will crash on that second pass.
    keep = _fsr_keep_fields()
    for field in required_input_fields or []:
        if field not in keep:
            keep.append(field)

    def cut_function(events: ak.Array) -> ak.Array:
        selected = baseline_main_preselection(
            events,
            lepton=lepton,
            pt_min=float(FSR_SETTINGS["PT_MIN"]),
            apply_medium_id=apply_medium_id,
            medium_field=medium_field,
        )
        return slim_keep_fields(selected, keep)

    return cut_function


def ensure_fsr_tight_parquet(lepton: str, backend: dict) -> Path:
    root = _fsr_root(lepton)
    if root.exists() and manifest_complete(root) and not FSR_SETTINGS["FORCE_REBUILD"]:
        log_step(f"[{lepton}] Reusing FSR tight parquet")
        return root

    log_step(f"[{lepton}] Building FSR tight parquet")
    reset_root_for_rebuild(root)

    build_fraction = float(FSR_SETTINGS["BUILD_FRACTION"])
    sample_rows: list[dict] = []
    subdirs: list[str] = []

    for sample_code in progress_iter(
        CHANNELS[lepton]["string_codes"],
        total=len(CHANNELS[lepton]["string_codes"]),
        desc=f"{lepton} fsr parquet",
        unit="sample",
    ):
        raw_fields = set(available_raw_fields(sample_code, backend))
        medium_field = choose_medium_field(sample_code, backend)
        apply_medium_id, reason = should_apply_medium_id(sample_code, medium_field)
        if not FSR_SETTINGS["APPLY_MEDIUM_ID"]:
            apply_medium_id = False
            reason = "disabled_in_fsr_settings"

        needed_raw = [
            field
            for field in [
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
            if field in raw_fields
        ]
        if medium_field is not None and medium_field not in needed_raw:
            needed_raw.append(medium_field)

        output_subdir = build_one_sample_to_root(
            sample_code=sample_code,
            root=root,
            read_vars=needed_raw,
            cut_function=_fsr_build_cut(
                lepton=lepton,
                apply_medium_id=apply_medium_id,
                medium_field=medium_field,
                required_input_fields=needed_raw,
            ),
            backend=backend,
            fraction=build_fraction,
        )
        sample_rows.append(
            {
                "sample": sample_code,
                "output_subdir": output_subdir,
                "medium_id_field": medium_field,
                "apply_medium_id": apply_medium_id,
                "apply_medium_id_reason": reason,
                "read_vars": needed_raw,
            }
        )
        if output_subdir is not None:
            subdirs.append(output_subdir)

    manifest = {
        "complete": True,
        "kind": "fsr_main",
        "channel": lepton,
        "created_at": now_stamp(),
        "settings": {
            "PT_MIN": FSR_SETTINGS["PT_MIN"],
            "BUILD_FRACTION": build_fraction,
            "APPLY_MEDIUM_ID": FSR_SETTINGS["APPLY_MEDIUM_ID"],
            "KEEP_FIELDS": _fsr_keep_fields(),
        },
        "subdirs": subdirs,
        "samples": sample_rows,
    }
    write_json(_fsr_manifest_path(root), manifest)
    return root


def load_fsr_events(lepton: str, sign: str, backend: dict) -> dict:
    root = ensure_fsr_tight_parquet(lepton, backend)
    manifest = read_manifest(root)
    if manifest is None:
        raise RuntimeError(f"Missing manifest under {root}")

    needed = _fsr_keep_fields() + ["weight", "totalWeight"]
    return load_tight_subdirs(
        root=root,
        subdirs=manifest["subdirs"],
        needed_fields=needed,
        backend=backend,
        fraction=float(FSR_SETTINGS["READ_FRACTION"]),
        cut_function=make_sign_cut(sign),
    )


def build_plot_dict_for_fsr(events_by_sample: dict, lepton: str) -> dict:
    """
    Keep the low-mass Drell-Yan sample as a background label so that the sigma calculation
    and the stacked plots tell the same story.
    """
    channel = CHANNELS[lepton]
    plot_dict: dict[str, ak.Array | None] = {}

    data_key = get_sample_key_by_prefix(events_by_sample, "2to4lep")
    plot_dict["Data"] = events_by_sample.get(data_key) if data_key else None

    primary_signal = channel["primary_signal"]
    primary_key = get_sample_key_by_prefix(events_by_sample, primary_signal)
    plot_dict[f"Signal {primary_signal}"] = events_by_sample.get(primary_key) if primary_key else None

    for sample_code in channel["signal_samples"]:
        if sample_code == primary_signal:
            continue
        sample_key = get_sample_key_by_prefix(events_by_sample, sample_code)
        plot_dict[f"Background {sample_code}"] = events_by_sample.get(sample_key) if sample_key else None

    for sample_code in channel["background_samples"]:
        sample_key = get_sample_key_by_prefix(events_by_sample, sample_code)
        plot_dict[f"Background {sample_code}"] = events_by_sample.get(sample_key) if sample_key else None

    return plot_dict


def add_alias_fields(events: ak.Array | None) -> ak.Array | None:
    if events is None:
        return None
    out = events
    if "mass_raw" not in out.fields:
        out = ak.with_field(out, out["mass"], "mass_raw")
    return out


def _make_p4(pt: ak.Array, eta: ak.Array, phi: ak.Array, energy: ak.Array):
    return vector.zip({"pt": pt, "eta": eta, "phi": phi, "E": energy})


def add_fsr_proxy_fields(events: ak.Array | None) -> ak.Array | None:
    if events is None:
        return None

    out = add_alias_fields(events)

    raw_mass = out["mass_raw"]
    etcone = out["lep_topoetcone20"]
    eta = out["lep_eta"]
    phi = out["lep_phi"]
    pt = out["lep_pt"]
    energy = out["lep_e"]

    apply_below = float(FSR_SETTINGS["FSR"]["APPLY_BELOW_MASS"])
    scale = float(FSR_SETTINGS["FSR"]["ETCONE_SCALE"])

    event_mask = raw_mass < apply_below
    event_mask_broadcast, _ = ak.broadcast_arrays(event_mask, pt)
    local_index = ak.local_index(etcone, axis=1)
    max_index = ak.argmax(etcone, axis=1, keepdims=True)

    for mode in FSR_SETTINGS["FSR"]["MODES"]:
        if mode == "both":
            dpt = scale * etcone
        elif mode == "maxcone":
            dpt = ak.where(local_index == max_index, scale * etcone, 0.0)
        else:
            raise ValueError(f"Unsupported FSR mode: {mode!r}")

        dpt = ak.where(event_mask_broadcast, dpt, 0.0)
        dE = dpt * np.cosh(eta)

        pt_corr = pt + dpt
        energy_corr = energy + dE

        p4_corr = _make_p4(pt_corr, eta, phi, energy_corr)
        mass_corr = (p4_corr[:, 0] + p4_corr[:, 1]).M
        delta_mass = mass_corr - raw_mass
        etcone_sub = ak.where(event_mask_broadcast, np.maximum(etcone - dpt, 0.0), etcone)

        out = ak.with_field(out, mass_corr, f"mass_fsr_{mode}")
        out = ak.with_field(out, delta_mass, f"delta_mass_fsr_{mode}")
        out = ak.with_field(out, etcone_sub, f"lep_topoetcone20_fsrsub_{mode}")

    return out


def enrich_plot_dict_with_fsr(plot_dict: dict) -> dict:
    return {label: add_fsr_proxy_fields(events) for label, events in plot_dict.items()}


def _both_or_leading_metric(events: ak.Array, field: str, require_both: bool) -> ak.Array:
    values = events[field]
    if require_both:
        return ak.max(values, axis=1)
    return values[:, 0]


def select_events(
    events: ak.Array | None,
    *,
    mass_window: tuple[float, float] | None = None,
    ptcone_max: float | None = None,
    etcone_max: float | None = None,
    require_both: bool = True,
    mass_field: str = "mass_raw",
    ptcone_field: str = "lep_ptvarcone30",
    etcone_field: str = "lep_topoetcone20",
) -> ak.Array | None:
    if events is None:
        return None

    if mass_field not in events.fields:
        raise KeyError(f"Missing mass field {mass_field!r}")
    if ptcone_field not in events.fields:
        raise KeyError(f"Missing ptcone field {ptcone_field!r}")
    if etcone_field not in events.fields:
        raise KeyError(f"Missing etcone field {etcone_field!r}")

    mask = ak.Array(np.ones(len(events), dtype=bool))

    if mass_window is not None:
        lo, hi = mass_window
        mask = mask & (events[mass_field] > float(lo)) & (events[mass_field] < float(hi))

    if ptcone_max is not None:
        pt_metric = _both_or_leading_metric(events, ptcone_field, require_both)
        mask = mask & (pt_metric < float(ptcone_max))

    if etcone_max is not None:
        et_metric = _both_or_leading_metric(events, etcone_field, require_both)
        mask = mask & (et_metric < float(etcone_max))

    return events[mask]


def select_plot_dict(
    plot_dict: dict,
    *,
    mass_window: tuple[float, float] | None = None,
    ptcone_max: float | None = None,
    etcone_max: float | None = None,
    require_both: bool = True,
    mass_field: str = "mass_raw",
    ptcone_field: str = "lep_ptvarcone30",
    etcone_field: str = "lep_topoetcone20",
) -> dict:
    return {
        label: select_events(
            events,
            mass_window=mass_window,
            ptcone_max=ptcone_max,
            etcone_max=etcone_max,
            require_both=require_both,
            mass_field=mass_field,
            ptcone_field=ptcone_field,
            etcone_field=etcone_field,
        )
        for label, events in plot_dict.items()
    }


def select_between_nominal_and_loose(
    plot_dict: dict,
    *,
    nominal_ptcone: float,
    nominal_etcone: float,
    loose_ptcone: float,
    loose_etcone: float,
    require_both: bool,
    mass_field: str = "mass_raw",
    ptcone_field: str = "lep_ptvarcone30",
    etcone_field: str = "lep_topoetcone20",
) -> dict:
    selected: dict[str, ak.Array | None] = {}
    for label, events in plot_dict.items():
        if events is None:
            selected[label] = None
            continue

        # Build both masks from the original events so the event bookkeeping stays trivial.
        pass_nominal_mask = selection_mask(
            events,
            ptcone_max=nominal_ptcone,
            etcone_max=nominal_etcone,
            require_both=require_both,
            mass_field=mass_field,
            ptcone_field=ptcone_field,
            etcone_field=etcone_field,
        )
        pass_loose_mask = selection_mask(
            events,
            ptcone_max=loose_ptcone,
            etcone_max=loose_etcone,
            require_both=require_both,
            mass_field=mass_field,
            ptcone_field=ptcone_field,
            etcone_field=etcone_field,
        )
        selected[label] = events[(~pass_nominal_mask) & pass_loose_mask]
    return selected


def selection_mask(
    events: ak.Array,
    *,
    mass_window: tuple[float, float] | None = None,
    ptcone_max: float | None = None,
    etcone_max: float | None = None,
    require_both: bool = True,
    mass_field: str = "mass_raw",
    ptcone_field: str = "lep_ptvarcone30",
    etcone_field: str = "lep_topoetcone20",
) -> ak.Array:
    mask = ak.Array(np.ones(len(events), dtype=bool))
    if mass_window is not None:
        lo, hi = mass_window
        mask = mask & (events[mass_field] > float(lo)) & (events[mass_field] < float(hi))
    if ptcone_max is not None:
        pt_metric = _both_or_leading_metric(events, ptcone_field, require_both)
        mask = mask & (pt_metric < float(ptcone_max))
    if etcone_max is not None:
        et_metric = _both_or_leading_metric(events, etcone_field, require_both)
        mask = mask & (et_metric < float(etcone_max))
    return mask


def _color_list(plot_dict: dict) -> list[str]:
    palette = ["k", "b", "olive", "g", "r", "m", "c", "orange"]
    return palette[: len(plot_dict)]


def save_stacked_mass_plot(
    *,
    backend: dict,
    plot_dict: dict,
    mass_field: str,
    output_path: Path,
    title: str,
    xmin: float,
    xmax: float,
    bins: int,
    logy: bool,
) -> None:
    figure, _ = backend["plot_stacked_hist"](
        plot_dict,
        mass_field,
        _color_list(plot_dict),
        bins,
        xmin,
        xmax,
        title,
        logy=logy,
        show_text=True,
        residual_plot=True,
        save_fig=False,
    )
    if FSR_SETTINGS["SAVE_PLOTS"]:
        save_fig(figure, output_path)


def summarize_plot_dict(plot_dict: dict) -> dict[str, float]:
    n_data = yield_data(plot_dict.get("Data"))
    n_signal = 0.0
    n_background = 0.0
    for label, events in plot_dict.items():
        if label == "Data":
            continue
        if label.startswith("Signal"):
            n_signal += yield_mc(events)
        else:
            n_background += yield_mc(events)
    return {
        "Data": n_data,
        "Signal": n_signal,
        "Background": n_background,
        "Data_minus_MC": n_data - (n_signal + n_background),
    }


def build_mass_window_table(plot_dict: dict, mass_field: str) -> pd.DataFrame:
    rows = []
    for low, high in FSR_SETTINGS["WINDOWS"]:
        selected = select_plot_dict(plot_dict, mass_window=(low, high), mass_field=mass_field)
        summary = summarize_plot_dict(selected)
        rows.append(
            {
                "mass_field": mass_field,
                "window": f"{low:.0f}-{high:.0f}",
                **summary,
            }
        )
    return pd.DataFrame(rows)


def compute_sigma_simple(
    *,
    plot_dict: dict,
    channel_config: dict,
    produced_event_count_fn,
    mass_window: tuple[float, float],
    ptcone_max: float,
    etcone_max: float,
    require_both: bool,
    mass_field: str,
    etcone_field: str = "lep_topoetcone20",
    produced_sumw_cache: dict[str, float] | None = None,
) -> dict:
    selected = select_plot_dict(
        plot_dict,
        mass_window=mass_window,
        ptcone_max=ptcone_max,
        etcone_max=etcone_max,
        require_both=require_both,
        mass_field=mass_field,
        etcone_field=etcone_field,
    )
    return compute_sigma_from_selected(
        selected_plot_os=selected,
        channel_config=channel_config,
        produced_event_count_fn=produced_event_count_fn,
        extra_bkg=0.0,
        produced_sumw_cache=produced_sumw_cache,
    )


def run_sigma_scan(
    *,
    plot_dict: dict,
    channel_config: dict,
    produced_event_count_fn,
    mass_window: tuple[float, float],
    nominal_ptcone: float,
    nominal_etcone: float,
    require_both: bool,
) -> pd.DataFrame:
    if not FSR_SETTINGS["SCAN"]["RUN"]:
        return pd.DataFrame()

    points = generate_scan_points(
        nominal_ptcone=nominal_ptcone,
        nominal_etcone=nominal_etcone,
        scan_mode=str(FSR_SETTINGS["SCAN"]["MODE"]),
        ptcone_range=tuple(FSR_SETTINGS["SCAN"]["PTCONE_RANGE"]),
        ptcone_step=float(FSR_SETTINGS["SCAN"]["PTCONE_STEP"]),
        etcone_range=tuple(FSR_SETTINGS["SCAN"]["ETCONE_RANGE"]),
        etcone_step=float(FSR_SETTINGS["SCAN"]["ETCONE_STEP"]),
        local_box_ptcone_half_width=float(FSR_SETTINGS["SCAN"]["LOCAL_BOX_PTCONE_HALF_WIDTH"]),
        local_box_etcone_half_width=float(FSR_SETTINGS["SCAN"]["LOCAL_BOX_ETCONE_HALF_WIDTH"]),
    )

    studies = {
        "raw": {"mass_field": "mass_raw", "etcone_field": "lep_topoetcone20"},
        "masscorr_maxcone": {"mass_field": "mass_fsr_maxcone", "etcone_field": "lep_topoetcone20"},
        "masscorr_both": {"mass_field": "mass_fsr_both", "etcone_field": "lep_topoetcone20"},
        # This one is the toy 'FSR-aware isolation' variant.
        "masscorr_etsub_maxcone": {
            "mass_field": "mass_fsr_maxcone",
            "etcone_field": "lep_topoetcone20_fsrsub_maxcone",
        },
    }

    cache: dict[str, float] = {}
    rows: list[dict] = []
    for ptcone_max, etcone_max in progress_iter(points, total=len(points), desc="fsr sigma scan", unit="pt"):
        row = {
            "ptcone_max": float(ptcone_max),
            "etcone_max": float(etcone_max),
            "is_nominal": bool(np.isclose(ptcone_max, nominal_ptcone) and np.isclose(etcone_max, nominal_etcone)),
        }
        for study_name, study_cfg in studies.items():
            try:
                result = compute_sigma_simple(
                    plot_dict=plot_dict,
                    channel_config=channel_config,
                    produced_event_count_fn=produced_event_count_fn,
                    mass_window=mass_window,
                    ptcone_max=float(ptcone_max),
                    etcone_max=float(etcone_max),
                    require_both=require_both,
                    mass_field=str(study_cfg["mass_field"]),
                    etcone_field=str(study_cfg["etcone_field"]),
                    produced_sumw_cache=cache,
                )
                row[f"sigma_pb_{study_name}"] = float(result["sigma_pb"])
                row[f"epsilon_{study_name}"] = float(result["epsilon"])
                row[f"n_sig_data_{study_name}"] = float(result["N_sig_data"])
            except Exception as exc:
                row[f"sigma_pb_{study_name}"] = np.nan
                row[f"epsilon_{study_name}"] = np.nan
                row[f"n_sig_data_{study_name}"] = np.nan
                row[f"error_{study_name}"] = str(exc)
        rows.append(row)

    scan_df = pd.DataFrame(rows)
    for study_name in studies:
        sigma_col = f"sigma_pb_{study_name}"
        if sigma_col not in scan_df.columns:
            continue
        nominal_rows = scan_df[scan_df["is_nominal"] & np.isfinite(scan_df[sigma_col])]
        if nominal_rows.empty:
            scan_df[f"sigma_abs_shift_pb_{study_name}"] = np.nan
            scan_df[f"sigma_frac_shift_{study_name}"] = np.nan
            continue
        nominal_sigma = float(nominal_rows.iloc[0][sigma_col])
        abs_shift = (scan_df[sigma_col] - nominal_sigma).abs()
        frac_shift = abs_shift / abs(nominal_sigma) if nominal_sigma != 0 else np.nan
        scan_df[f"sigma_abs_shift_pb_{study_name}"] = abs_shift
        scan_df[f"sigma_frac_shift_{study_name}"] = frac_shift
    return scan_df


def save_scan_outputs(scan_df: pd.DataFrame, output_dir: Path, nominal_ptcone: float, nominal_etcone: float) -> None:
    if scan_df.empty:
        return
    scan_df.to_csv(output_dir / "sigma_scan.csv", index=False)
    write_json(output_dir / "sigma_scan.json", scan_df.to_dict(orient="records"))

    for study_name, title in [
        ("raw", "sigma (raw mass, raw etcone)"),
        ("masscorr_maxcone", "sigma (mass corrected: maxcone)"),
        ("masscorr_both", "sigma (mass corrected: both)"),
        ("masscorr_etsub_maxcone", "sigma (mass corrected + etcone-subtracted: maxcone)"),
    ]:
        sigma_col = f"sigma_pb_{study_name}"
        if sigma_col not in scan_df.columns:
            continue
        save_scan_heatmap(
            scan_df,
            output_dir / f"heatmap_{study_name}.png",
            value_column=sigma_col,
            value_label="sigma_pb [pb]",
            title=title,
            nominal_ptcone=nominal_ptcone,
            nominal_etcone=nominal_etcone,
        )
        shift_col = f"sigma_frac_shift_{study_name}"
        if shift_col in scan_df.columns:
            save_scan_heatmap(
                scan_df,
                output_dir / f"heatmap_fracshift_{study_name}.png",
                value_column=shift_col,
                value_label="fractional |Δsigma|",
                title=f"fractional |Δsigma| ({study_name})",
                nominal_ptcone=nominal_ptcone,
                nominal_etcone=nominal_etcone,
            )


def run() -> Path:
    ensure_script_directory()
    ensure_environment()
    backend = import_backend()

    lepton = str(FSR_SETTINGS["LEPTON"]).strip().lower()
    if lepton != "mu":
        raise ValueError("This focused FSR study is currently implemented for the muon channel only.")

    run_root = (Path(__file__).resolve().parent / FSR_SETTINGS["OUTPUT_DIR"] / f"run_{now_stamp()}").resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(run_root / "fsr_settings.json", FSR_SETTINGS)

    nominal_ptcone = float(FSR_SETTINGS["NOMINAL_ISO"]["ptcone_max"])
    nominal_etcone = float(FSR_SETTINGS["NOMINAL_ISO"]["etcone_max"])
    loose_ptcone = float(FSR_SETTINGS["LOOSE_ISO"]["ptcone_max"])
    loose_etcone = float(FSR_SETTINGS["LOOSE_ISO"]["etcone_max"])
    mass_window = tuple(FSR_SETTINGS["MASS_WINDOW"])
    require_both = bool(FSR_SETTINGS["REQUIRE_BOTH_ISO"])

    data_os = load_fsr_events(lepton, "OS", backend)
    plot_os = enrich_plot_dict_with_fsr(build_plot_dict_for_fsr(data_os, lepton))

    # 1) Main control sample: fail nominal but pass loose.
    between_raw = select_between_nominal_and_loose(
        plot_os,
        nominal_ptcone=nominal_ptcone,
        nominal_etcone=nominal_etcone,
        loose_ptcone=loose_ptcone,
        loose_etcone=loose_etcone,
        require_both=require_both,
        mass_field="mass_raw",
    )

    plots_dir = run_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for plot_name, plot_dict, mass_field, cfg_key, title in [
        (
            "between_raw_full",
            between_raw,
            "mass_raw",
            "FULL",
            f"OS dimuon: fail nominal ({nominal_ptcone:.2f},{nominal_etcone:.2f}) but pass loose ({loose_ptcone:.2f},{loose_etcone:.2f}) [raw mass]",
        ),
        (
            "between_fsr_maxcone_full",
            between_raw,
            "mass_fsr_maxcone",
            "FULL",
            f"Same control sample with toy FSR recovery [maxcone mode]",
        ),
        (
            "between_fsr_both_full",
            between_raw,
            "mass_fsr_both",
            "FULL",
            f"Same control sample with toy FSR recovery [both-muons mode]",
        ),
        (
            "between_raw_zoom",
            between_raw,
            "mass_raw",
            "ZOOM",
            f"Control sample near Z peak [raw mass]",
        ),
        (
            "between_fsr_maxcone_zoom",
            between_raw,
            "mass_fsr_maxcone",
            "ZOOM",
            f"Control sample near Z peak [FSR recovery: maxcone]",
        ),
        (
            "between_fsr_both_zoom",
            between_raw,
            "mass_fsr_both",
            "ZOOM",
            f"Control sample near Z peak [FSR recovery: both]",
        ),
    ]:
        plot_cfg = FSR_SETTINGS["MASS_PLOT"][cfg_key]
        save_stacked_mass_plot(
            backend=backend,
            plot_dict=plot_dict,
            mass_field=mass_field,
            output_path=plots_dir / f"{plot_name}.png",
            title=title,
            xmin=float(plot_cfg["xmin"]),
            xmax=float(plot_cfg["xmax"]),
            bins=int(plot_cfg["bins"]),
            logy=bool(plot_cfg["logy"]),
        )

    # 2) Also inspect the full pass-loose region because that is closer to the sigma scan story.
    pass_loose_raw = select_plot_dict(
        plot_os,
        ptcone_max=loose_ptcone,
        etcone_max=loose_etcone,
        require_both=require_both,
        mass_field="mass_raw",
    )
    for plot_name, mass_field, cfg_key, title in [
        (
            "pass_loose_raw_zoom",
            "mass_raw",
            "ZOOM",
            f"Pass loose isolation ({loose_ptcone:.2f},{loose_etcone:.2f}) [raw mass]",
        ),
        (
            "pass_loose_fsr_maxcone_zoom",
            "mass_fsr_maxcone",
            "ZOOM",
            f"Pass loose isolation with toy FSR recovery [maxcone]",
        ),
        (
            "pass_loose_fsr_both_zoom",
            "mass_fsr_both",
            "ZOOM",
            f"Pass loose isolation with toy FSR recovery [both]",
        ),
    ]:
        plot_cfg = FSR_SETTINGS["MASS_PLOT"][cfg_key]
        save_stacked_mass_plot(
            backend=backend,
            plot_dict=pass_loose_raw,
            mass_field=mass_field,
            output_path=plots_dir / f"{plot_name}.png",
            title=title,
            xmin=float(plot_cfg["xmin"]),
            xmax=float(plot_cfg["xmax"]),
            bins=int(plot_cfg["bins"]),
            logy=bool(plot_cfg["logy"]),
        )

    # 3) Save compact window tables for quick interpretation.
    windows_raw = build_mass_window_table(between_raw, "mass_raw")
    windows_maxcone = build_mass_window_table(between_raw, "mass_fsr_maxcone")
    windows_both = build_mass_window_table(between_raw, "mass_fsr_both")
    window_table = pd.concat([windows_raw, windows_maxcone, windows_both], ignore_index=True)
    window_table.to_csv(run_root / "between_mass_windows.csv", index=False)

    # 4) Sigma scan (simple cut-and-count, no extra DD term) to see whether the toy correction
    #    changes the isolation dependence pattern.
    scan_df = run_sigma_scan(
        plot_dict=plot_os,
        channel_config=CHANNELS[lepton],
        produced_event_count_fn=backend["produced_event_count"],
        mass_window=mass_window,
        nominal_ptcone=nominal_ptcone,
        nominal_etcone=nominal_etcone,
        require_both=require_both,
    )
    scan_dir = run_root / "sigma_scan"
    scan_dir.mkdir(parents=True, exist_ok=True)
    save_scan_outputs(scan_df, scan_dir, nominal_ptcone, nominal_etcone)

    # 5) Summaries.
    summary_lines = [
        f"FSR toy study run directory: {run_root}",
        "",
        "Physics reminder:",
        "- mass_fsr_maxcone: add topoetcone20 back only to the muon with larger topoetcone20,",
        "  and only when raw dimuon mass is below APPLY_BELOW_MASS.",
        "- masscorr_etsub_maxcone heatmap also subtracts that recovered proxy from the etcone",
        "  used in the selection, so it is the most direct toy test of an FSR-aware isolation idea.",
        "",
        "Nominal / loose isolation:",
        f"- nominal: ptcone < {nominal_ptcone:.3f}, etcone < {nominal_etcone:.3f}",
        f"- loose:   ptcone < {loose_ptcone:.3f}, etcone < {loose_etcone:.3f}",
        "",
        "Control sample summaries (fail nominal, pass loose):",
        f"raw mass yields:      {summarize_plot_dict(between_raw)}",
        f"FSR maxcone yields:   {summarize_plot_dict(select_plot_dict(between_raw, mass_field='mass_fsr_maxcone'))}",
        f"FSR both yields:      {summarize_plot_dict(select_plot_dict(between_raw, mass_field='mass_fsr_both'))}",
        "",
        "Key files to inspect first:",
        f"- {plots_dir / 'between_raw_zoom.png'}",
        f"- {plots_dir / 'between_fsr_maxcone_zoom.png'}",
        f"- {plots_dir / 'between_fsr_both_zoom.png'}",
        f"- {scan_dir / 'heatmap_raw.png'}",
        f"- {scan_dir / 'heatmap_masscorr_maxcone.png'}",
        f"- {scan_dir / 'heatmap_masscorr_etsub_maxcone.png'}",
        "",
        "Interpretation hints:",
        "1. If the low-mass excess migrates upward toward the Z peak after the toy recovery,",
        "   that supports the radiative / FSR-like explanation.",
        "2. If the raw and corrected sigma heatmaps look similarly monotonic, then a mass-only",
        "   recovery is not enough to explain the cut dependence.",
        "3. If the masscorr_etsub_maxcone heatmap is visibly flatter than the raw one, that is",
        "   a sign that the etcone-driven drift is plausibly coming from near-muon radiated energy.",
    ]
    write_text(run_root / "summary.txt", "\n".join(summary_lines))

    return run_root


if __name__ == "__main__":
    output = run()
    print(f"FSR study outputs saved under: {output}")
