from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import awkward as ak
import math
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from config import DD_METHODS, LUMI_REL_UNC, SETTINGS
from cross_section import compute_sigma_from_selected
from parquet_io import (
    _chunk_read_columns_for_control,
    accumulate_control_stage_totals_from_tight_chunks,
    collect_channel_control_samples,
    ensure_control_tight_parquet,
    read_manifest,
)
from selections import apply_selection
from utils import infer_sample_code_from_name, log_step, progress_iter, weight_field, write_json, write_text, yield_mc
from visualisation import (
    cutflow_table,
    save_additional_bkg_plot,
    save_estimator_plot,
    select_plotdict,
    summarize,
)


DEBUG_REGION_NAMES = ["ordered_11_13", "ordered_13_11", "ep_mum", "mup_em", "ee_ss", "mumu_ss"]
PHYSICAL_REGION_NAMES = ["ep_mum", "mup_em", "ee_ss", "mumu_ss"]
FINAL_STAGE_NAME = "mass_plus_iso"
STAGE_LABELS = {
    "baseline": "baseline control preselection",
    "mass_only": "mass-only selection",
    FINAL_STAGE_NAME: "mass + iso selection",
}


def ordered_wrong_flavour_masks(events: ak.Array) -> dict[str, ak.Array]:
    type0 = events["lep_type"][:, 0]
    type1 = events["lep_type"][:, 1]
    return {
        "ordered_11_13": (type0 == 11) & (type1 == 13),
        "ordered_13_11": (type0 == 13) & (type1 == 11),
    }


def control_region_masks(events: ak.Array) -> dict[str, ak.Array]:
    type0 = events["lep_type"][:, 0]
    type1 = events["lep_type"][:, 1]
    charge0 = events["lep_charge"][:, 0]
    charge1 = events["lep_charge"][:, 1]
    charge_product = charge0 * charge1

    # These are physical charge/flavour categories, not index-ordered regions.
    ep_mum = (
        ((type0 == 11) & (charge0 > 0) & (type1 == 13) & (charge1 < 0))
        | ((type1 == 11) & (charge1 > 0) & (type0 == 13) & (charge0 < 0))
    )
    mup_em = (
        ((type0 == 13) & (charge0 > 0) & (type1 == 11) & (charge1 < 0))
        | ((type1 == 13) & (charge1 > 0) & (type0 == 11) & (charge0 < 0))
    )

    return {
        "ep_mum": ep_mum,
        "mup_em": mup_em,
        "ee_ss": (type0 == 11) & (type1 == 11) & (charge_product > 0),
        "mumu_ss": (type0 == 13) & (type1 == 13) & (charge_product > 0),
    }


def all_control_region_masks(events: ak.Array) -> dict[str, ak.Array]:
    masks = ordered_wrong_flavour_masks(events)
    masks.update(control_region_masks(events))
    return masks


def summarise_control_regions(events: ak.Array | None) -> dict[str, dict[str, float]]:
    if events is None:
        zero_counts = {region: 0.0 for region in DEBUG_REGION_NAMES}
        return {"count": dict(zero_counts), "weight": dict(zero_counts)}

    masks = all_control_region_masks(events)
    return {
        "count": {region: float(len(events[masks[region]])) for region in DEBUG_REGION_NAMES},
        "weight": {region: float(yield_mc(events[masks[region]])) for region in DEBUG_REGION_NAMES},
    }


def clip_value(value: float, clip_negative: bool) -> float:
    return max(float(value), 0.0) if clip_negative else float(value)


def wrong_charge_region_name(lepton: str) -> str:
    if lepton == "mu":
        return "mumu_ss"
    if lepton == "e":
        return "ee_ss"
    raise ValueError(f"Unsupported channel for wrong-charge estimator: {lepton!r}")


def _stage_selectors(
    mass_window: tuple[float, float],
    ptcone_max: float,
    etcone_max: float,
    require_both: bool,
):
    return {
        "baseline": lambda events: events,
        "mass_only": lambda events: apply_selection(events, mass_window=mass_window, require_both=require_both),
        FINAL_STAGE_NAME: lambda events: apply_selection(
            events,
            mass_window=mass_window,
            ptcone_max=ptcone_max,
            etcone_max=etcone_max,
            require_both=require_both,
        ),
    }


def _cache_key(
    lepton: str,
    mass_window: tuple[float, float],
    ptcone_max: float,
    etcone_max: float,
    require_both: bool,
) -> tuple:
    return (
        lepton,
        round(float(mass_window[0]), 6),
        round(float(mass_window[1]), 6),
        round(float(ptcone_max), 6),
        round(float(etcone_max), 6),
        bool(require_both),
    )


def _resolved_methods(methods: Sequence[str] | None) -> tuple[str, ...]:
    if methods is None:
        return DD_METHODS

    ordered: list[str] = []
    for method in methods:
        if method not in DD_METHODS:
            raise ValueError(f"Unsupported DD method {method!r}")
        if method not in ordered:
            ordered.append(method)
    if not ordered:
        raise ValueError("At least one DD method must be requested")
    return tuple(ordered)


def _final_selection_mask(
    events: ak.Array,
    mass_window: tuple[float, float],
    ptcone_max: float,
    etcone_max: float,
    require_both: bool,
) -> np.ndarray:
    masses = np.asarray(events["mass"], dtype=float)
    if require_both:
        ptcone_metric = np.maximum(
            np.asarray(events["lep_ptvarcone30"][:, 0], dtype=float),
            np.asarray(events["lep_ptvarcone30"][:, 1], dtype=float),
        )
        etcone_metric = np.maximum(
            np.asarray(events["lep_topoetcone20"][:, 0], dtype=float),
            np.asarray(events["lep_topoetcone20"][:, 1], dtype=float),
        )
    else:
        ptcone_metric = np.asarray(events["lep_ptvarcone30"][:, 0], dtype=float)
        etcone_metric = np.asarray(events["lep_topoetcone20"][:, 0], dtype=float)

    return (
        (masses > float(mass_window[0]))
        & (masses < float(mass_window[1]))
        & (ptcone_metric < float(ptcone_max))
        & (etcone_metric < float(etcone_max))
    )


def _chunk_weights(events: ak.Array) -> np.ndarray:
    field = weight_field(events)
    if field is None:
        return np.ones(len(events), dtype=float)
    return np.asarray(events[field], dtype=float)


def gather_control_final_totals_for_points(
    backend: dict,
    lepton: str,
    mass_window: tuple[float, float],
    cut_points: Sequence[tuple[float, float]],
    require_both: bool,
    cache: dict | None = None,
) -> dict[tuple, dict]:
    point_map: dict[tuple, tuple[float, float]] = {}
    totals_by_key: dict[tuple, dict] = {}

    for ptcone_max, etcone_max in cut_points:
        key = _cache_key(lepton, mass_window, ptcone_max, etcone_max, require_both)
        if key in totals_by_key or key in point_map:
            continue
        if cache is not None and key in cache:
            totals_by_key[key] = cache[key]
            continue
        point_map[key] = (float(ptcone_max), float(etcone_max))

    if not point_map:
        return totals_by_key

    root = ensure_control_tight_parquet(backend)
    manifest = read_manifest(root)
    if manifest is None:
        raise RuntimeError(f"Missing control manifest under {root}")

    sample_info = collect_channel_control_samples(lepton)
    data_sample = sample_info["data_sample"]
    mc_sample_set = set(sample_info["mc_samples"])
    relevant_samples = set(sample_info["all_samples"])

    subdir_to_sample = {}
    for row in manifest.get("samples", []):
        if row.get("output_subdir") and row.get("sample"):
            subdir_to_sample[str(row["output_subdir"])] = str(row["sample"])

    for key in point_map:
        totals_by_key[key] = {
            "data_counts": {region: 0.0 for region in PHYSICAL_REGION_NAMES},
            "mc_counts": {region: 0.0 for region in PHYSICAL_REGION_NAMES},
            "mc_samples_included": set(),
        }

    relevant_subdirs = []
    for subdir_name in manifest["subdirs"]:
        sample_code = subdir_to_sample.get(subdir_name)
        if sample_code is None:
            sample_code = infer_sample_code_from_name(subdir_name, list(relevant_samples))
        if sample_code is None or sample_code not in relevant_samples:
            continue
        relevant_subdirs.append((subdir_name, sample_code))

    if point_map:
        log_step(f"[{lepton}] Accumulating scan control-region yields in one pass")

    iterator = progress_iter(relevant_subdirs, total=len(relevant_subdirs), desc=f"{lepton} scan control", unit="sample")
    for subdir_name, sample_code in iterator:
        subdir = root / subdir_name
        for parquet_file in sorted(subdir.rglob("*.parquet")):
            parquet_handle = pq.ParquetFile(parquet_file)
            file_fields = set(parquet_handle.schema_arrow.names)
            missing = {"lep_type", "lep_charge", "lep_ptvarcone30", "lep_topoetcone20", "mass"} - file_fields
            if missing:
                raise KeyError(f"Missing required control fields in {parquet_file}: {sorted(missing)}")

            columns = _chunk_read_columns_for_control(file_fields)
            for row_group in range(parquet_handle.num_row_groups):
                table = parquet_handle.read_row_group(row_group, columns=columns)
                if table.num_rows == 0:
                    continue

                chunk = ak.from_arrow(table)
                region_masks = {
                    region: np.asarray(mask, dtype=bool)
                    for region, mask in control_region_masks(chunk).items()
                }
                chunk_weights = _chunk_weights(chunk)

                for key, (ptcone_max, etcone_max) in point_map.items():
                    final_mask = _final_selection_mask(
                        chunk,
                        mass_window=mass_window,
                        ptcone_max=ptcone_max,
                        etcone_max=etcone_max,
                        require_both=require_both,
                    )
                    if not np.any(final_mask):
                        continue

                    point_totals = totals_by_key[key]
                    if sample_code == data_sample:
                        for region in PHYSICAL_REGION_NAMES:
                            point_totals["data_counts"][region] += float(np.count_nonzero(final_mask & region_masks[region]))
                    elif sample_code in mc_sample_set:
                        for region in PHYSICAL_REGION_NAMES:
                            combined_mask = final_mask & region_masks[region]
                            if np.any(combined_mask):
                                point_totals["mc_counts"][region] += float(np.sum(chunk_weights[combined_mask]))
                        point_totals["mc_samples_included"].add(sample_code)

    for key, point_totals in totals_by_key.items():
        point_totals["mc_samples_included"] = sorted(point_totals["mc_samples_included"])
        if cache is not None:
            cache[key] = point_totals

    return totals_by_key


def gather_control_stage_totals(
    backend: dict,
    lepton: str,
    mass_window: tuple[float, float],
    ptcone_max: float,
    etcone_max: float,
    require_both: bool,
    cache: dict | None = None,
) -> dict:
    key = _cache_key(lepton, mass_window, ptcone_max, etcone_max, require_both)
    if cache is not None and key in cache:
        return cache[key]

    totals = accumulate_control_stage_totals_from_tight_chunks(
        backend=backend,
        lepton=lepton,
        stage_selectors=_stage_selectors(mass_window, ptcone_max, etcone_max, require_both),
        stage_region_summariser=summarise_control_regions,
        region_names=DEBUG_REGION_NAMES,
    )
    if cache is not None:
        cache[key] = totals
    return totals


def control_stage_totals_to_frame(stage_totals: dict) -> pd.DataFrame:
    rows = []
    for stage_name, stage_result in stage_totals.items():
        for region in DEBUG_REGION_NAMES:
            n_data = stage_result["data_counts"][region]
            n_mc = stage_result["mc_counts"][region]
            rows.append(
                {
                    "stage": stage_name,
                    "stage_label": STAGE_LABELS[stage_name],
                    "region": region,
                    "N_data": n_data,
                    "N_MC": n_mc,
                    "residual": n_data - n_mc,
                }
            )
    return pd.DataFrame(rows)


def index_ordering_diagnostic(stage_totals: dict) -> str:
    lines = []
    detected = False
    for stage_name in ("baseline", "mass_only", FINAL_STAGE_NAME):
        stage_result = stage_totals[stage_name]
        ordered_11_13 = stage_result["data_counts"]["ordered_11_13"]
        ordered_13_11 = stage_result["data_counts"]["ordered_13_11"]
        physical_total = stage_result["data_counts"]["ep_mum"] + stage_result["data_counts"]["mup_em"]
        if physical_total > 0 and (ordered_11_13 == 0.0 or ordered_13_11 == 0.0):
            detected = True
            lines.append(
                f"{STAGE_LABELS[stage_name]}: ordered opposite-flavour counts are one-sided "
                f"((11,13)={ordered_11_13:.3f}, (13,11)={ordered_13_11:.3f}) while the physical "
                "wrong-flavour regions remain populated. This is consistent with fixed ntuple lepton ordering."
            )
        else:
            lines.append(
                f"{STAGE_LABELS[stage_name]}: no strong one-sided ordered-pair population was seen "
                f"((11,13)={ordered_11_13:.3f}, (13,11)={ordered_13_11:.3f})."
            )

    if detected:
        lines.insert(
            0,
            "An index-ordering issue is present: ordered opposite-flavour counts are asymmetric even when physical wrong-flavour categories survive.",
        )
    else:
        lines.insert(0, "No strong index-ordering issue was detected in the checked control-region stages.")
    return "\n".join(lines)


def save_control_region_debug_outputs(lepton: str, stage_totals: dict, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    table = control_stage_totals_to_frame(stage_totals)
    summary_text = index_ordering_diagnostic(stage_totals)

    table.to_csv(output_dir / f"{lepton}_control_region_debug.csv", index=False)
    write_json(
        output_dir / f"{lepton}_control_region_debug.json",
        {"channel": lepton, "stage_totals": stage_totals, "summary": summary_text},
    )
    write_text(output_dir / f"{lepton}_control_region_debug_summary.txt", summary_text + "\n\n" + table.to_string(index=False))
    return {"table": table, "summary": summary_text}


def compute_estimator_tables(
    lepton: str,
    data_counts: dict[str, float],
    mc_counts: dict[str, float],
    clip_negative: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float | str]]]:
    residuals = {region: data_counts[region] - mc_counts[region] for region in PHYSICAL_REGION_NAMES}
    clipped_residuals = {region: clip_value(value, clip_negative) for region, value in residuals.items()}

    regions_table = pd.DataFrame(
        [
            {
                "region": region,
                "N_data": data_counts[region],
                "N_MC": mc_counts[region],
                "residual": residuals[region],
                "clipped_residual": clipped_residuals[region],
            }
            for region in PHYSICAL_REGION_NAMES
        ]
    )

    wrong_flavour_raw = 0.5 * (residuals["ep_mum"] + residuals["mup_em"])
    wrong_charge_region = wrong_charge_region_name(lepton)
    wrong_charge_raw = residuals[wrong_charge_region]
    both_average_raw = 0.5 * (wrong_flavour_raw + wrong_charge_raw)

    estimator_map: dict[str, dict[str, float | str]] = {
        "none": {"description": "Baseline with no additional DD background", "raw": 0.0},
        "wrong_flavour": {"description": "Average wrong-flavour residual", "raw": wrong_flavour_raw},
        "wrong_charge": {"description": f"Wrong-charge residual from {wrong_charge_region}", "raw": wrong_charge_raw},
        "both_average": {"description": "Average of wrong_flavour and wrong_charge", "raw": both_average_raw},
    }
    for method in estimator_map:
        estimator_map[method]["clipped"] = clip_value(estimator_map[method]["raw"], clip_negative)

    estimators_table = pd.DataFrame(
        [
            {
                "method": method,
                "estimate_raw": estimator_map[method]["raw"],
                "estimate_clipped": estimator_map[method]["clipped"],
                "description": estimator_map[method]["description"],
            }
            for method in DD_METHODS
        ]
    )
    return regions_table, estimators_table, estimator_map


def _final_stage_counts(stage_totals: dict) -> tuple[dict[str, float], dict[str, float]]:
    final_stage = stage_totals[FINAL_STAGE_NAME]
    data_counts = {region: final_stage["data_counts"][region] for region in PHYSICAL_REGION_NAMES}
    mc_counts = {region: final_stage["mc_counts"][region] for region in PHYSICAL_REGION_NAMES}
    return data_counts, mc_counts


def build_sigma_results_table(
    plot_os: dict,
    channel_config: dict,
    produced_event_count_fn,
    mass_window: tuple[float, float],
    ptcone_max: float,
    etcone_max: float,
    require_both: bool,
    estimator_map: dict[str, dict[str, float | str]],
    produced_cache: dict[str, float] | None,
    methods: Sequence[str] | None = None,
    selected_os: dict | None = None,
) -> pd.DataFrame:
    resolved_methods = _resolved_methods(methods)
    if selected_os is None:
        selected_os = select_plotdict(
            plot_os,
            mass_window=mass_window,
            ptcone_max=ptcone_max,
            etcone_max=etcone_max,
            require_both=require_both,
        )

    rows = []
    sigma_none = None
    sigma_none_valid = False
    for method in resolved_methods:
        extra_background = 0.0 if method == "none" else float(estimator_map[method]["clipped"])
        sigma_valid = True
        sigma_error = ""
        try:
            sigma_result = compute_sigma_from_selected(
                selected_plot_os=selected_os,
                channel_config=channel_config,
                produced_event_count_fn=produced_event_count_fn,
                extra_bkg=extra_background,
                produced_sumw_cache=produced_cache,
            )
        except ZeroDivisionError as exc:
            sigma_valid = False
            sigma_error = str(exc)
            sigma_result = {
                "sigma_pb": math.nan,
                "dsigma_stat_pb": math.nan,
                "dsigma_lumi_pb": math.nan,
                "epsilon": math.nan,
                "N_selected": math.nan,
                "N_bkg_total": math.nan,
                "N_sig_data": math.nan,
            }
        if method == "none" and sigma_valid:
            sigma_none = sigma_result["sigma_pb"]
            sigma_none_valid = True

        sigma_shift_pb = math.nan
        if sigma_valid and sigma_none_valid and sigma_none is not None:
            sigma_shift_pb = sigma_result["sigma_pb"] - float(sigma_none)
        rows.append(
            {
                "method": method,
                "extra_bkg": extra_background,
                "sigma_pb": sigma_result["sigma_pb"],
                "sigma_shift_pb": sigma_shift_pb,
                "dsigma_stat_pb": sigma_result["dsigma_stat_pb"],
                "dsigma_lumi_pb": sigma_result["dsigma_lumi_pb"],
                "epsilon": sigma_result["epsilon"],
                "N_selected": sigma_result["N_selected"],
                "N_bkg_total": sigma_result["N_bkg_total"],
                "N_sig_data": sigma_result["N_sig_data"],
                "sigma_valid": sigma_valid,
                "sigma_error": sigma_error,
            }
        )
    return pd.DataFrame(rows)


def evaluate_cut_point(
    lepton: str,
    plot_os: dict,
    channel_config: dict,
    produced_event_count_fn,
    backend: dict | None,
    mass_window: tuple[float, float],
    ptcone_max: float,
    etcone_max: float,
    require_both: bool,
    plot_ss: dict | None = None,
    control_cache: dict | None = None,
    produced_cache: dict[str, float] | None = None,
    stage_totals: dict | None = None,
    include_cutflows: bool = False,
    methods: Sequence[str] | None = None,
) -> dict:
    resolved_methods = _resolved_methods(methods)
    if stage_totals is None:
        if backend is None:
            raise ValueError("backend is required when stage_totals are not supplied")
        stage_totals = gather_control_stage_totals(
            backend=backend,
            lepton=lepton,
            mass_window=mass_window,
            ptcone_max=ptcone_max,
            etcone_max=etcone_max,
            require_both=require_both,
            cache=control_cache,
        )

    selected_os = select_plotdict(
        plot_os,
        mass_window=mass_window,
        ptcone_max=ptcone_max,
        etcone_max=etcone_max,
        require_both=require_both,
    )
    os_data, os_signal, os_background, os_s_over_b, os_data_over_mc = summarize(selected_os)
    signal_region_yields = {
        "Data": os_data,
        "Signal": os_signal,
        "Background": os_background,
        "S/B": os_s_over_b,
        "Data/MC": os_data_over_mc,
    }

    ss_yields = None
    if plot_ss is not None:
        selected_ss = select_plotdict(
            plot_ss,
            mass_window=mass_window,
            ptcone_max=ptcone_max,
            etcone_max=etcone_max,
            require_both=require_both,
        )
        ss_data, ss_signal, ss_background, ss_s_over_b, ss_data_over_mc = summarize(selected_ss)
        ss_yields = {
            "Data": ss_data,
            "Signal": ss_signal,
            "Background": ss_background,
            "S/B": ss_s_over_b,
            "Data/MC": ss_data_over_mc,
        }

    data_counts, mc_counts = _final_stage_counts(stage_totals)
    regions_table, estimators_table, estimator_map = compute_estimator_tables(
        lepton=lepton,
        data_counts=data_counts,
        mc_counts=mc_counts,
        clip_negative=bool(SETTINGS["ADDITIONAL_DATA_DRIVEN_BKG"]["CLIP_NEGATIVE_TO_ZERO"]),
    )
    sigma_results_table = build_sigma_results_table(
        plot_os=plot_os,
        channel_config=channel_config,
        produced_event_count_fn=produced_event_count_fn,
        mass_window=mass_window,
        ptcone_max=ptcone_max,
        etcone_max=etcone_max,
        require_both=require_both,
        estimator_map=estimator_map,
        produced_cache=produced_cache,
        methods=resolved_methods,
        selected_os=selected_os,
    )
    sigma_results_table["dsigma_lumi_pb"] = sigma_results_table["sigma_pb"] * LUMI_REL_UNC
    comparison_sigma_if_applied = sigma_results_table[sigma_results_table["method"] != "none"].copy()
    if "none" in sigma_results_table["method"].tolist():
        sigma_none = float(sigma_results_table.loc[sigma_results_table["method"] == "none", "sigma_pb"].iloc[0])
        comparison_sigma_if_applied["sigma_shift_pb"] = comparison_sigma_if_applied["sigma_pb"] - sigma_none

    os_cutflow = None
    ss_cutflow = None
    if include_cutflows:
        os_cutflow = cutflow_table(
            plot_os,
            mass_window=mass_window,
            ptcone_max=ptcone_max,
            etcone_max=etcone_max,
            require_both=require_both,
        )
        if plot_ss is not None:
            ss_cutflow = cutflow_table(
                plot_ss,
                mass_window=mass_window,
                ptcone_max=ptcone_max,
                etcone_max=etcone_max,
                require_both=require_both,
            )

    return {
        "channel": lepton,
        "cuts": {
            "mass_window": list(mass_window),
            "ptcone_max": float(ptcone_max),
            "etcone_max": float(etcone_max),
            "require_both_iso": bool(require_both),
        },
        "signal_region_yields": signal_region_yields,
        "ss_yields": ss_yields,
        "stage_totals": stage_totals,
        "regions_table": regions_table,
        "estimators_table": estimators_table,
        "sigma_results_table": sigma_results_table,
        "comparison_sigma_if_applied": comparison_sigma_if_applied,
        "os_cutflow": os_cutflow,
        "ss_cutflow": ss_cutflow,
        "debug_summary": index_ordering_diagnostic(stage_totals),
    }


def evaluate_scan_cut_points(
    lepton: str,
    plot_os: dict,
    channel_config: dict,
    produced_event_count_fn,
    backend: dict,
    mass_window: tuple[float, float],
    cut_points: Sequence[tuple[float, float]],
    require_both: bool,
    nominal_cut_point: dict | None = None,
    produced_cache: dict[str, float] | None = None,
    control_cache: dict | None = None,
    methods: Sequence[str] | None = None,
) -> dict[tuple[float, float], dict]:
    resolved_methods = _resolved_methods(methods)
    unique_points: dict[tuple, tuple[float, float]] = {}
    for ptcone_max, etcone_max in cut_points:
        key = _cache_key(lepton, mass_window, ptcone_max, etcone_max, require_both)
        if key not in unique_points:
            unique_points[key] = (float(ptcone_max), float(etcone_max))

    if not unique_points:
        return {}

    nominal_key = None
    nominal_lookup = None
    if nominal_cut_point is not None:
        nominal_cuts = nominal_cut_point["cuts"]
        nominal_key = _cache_key(
            lepton,
            tuple(nominal_cuts["mass_window"]),
            float(nominal_cuts["ptcone_max"]),
            float(nominal_cuts["etcone_max"]),
            bool(nominal_cuts["require_both_iso"]),
        )
        nominal_lookup = nominal_cut_point["sigma_results_table"].set_index("method")

    batched_control_totals = gather_control_final_totals_for_points(
        backend=backend,
        lepton=lepton,
        mass_window=mass_window,
        cut_points=list(unique_points.values()),
        require_both=require_both,
        cache=control_cache,
    )

    evaluations: dict[tuple[float, float], dict] = {}
    iterator = progress_iter(unique_points.items(), total=len(unique_points), desc=f"{lepton} scan sigma", unit="pt")
    for key, (ptcone_max, etcone_max) in iterator:
        if nominal_key is not None and key == nominal_key and nominal_lookup is not None:
            sigma_results_table = (
                nominal_lookup.loc[list(resolved_methods)]
                .reset_index()
                .copy()
            )
            evaluations[(ptcone_max, etcone_max)] = {
                "sigma_results_table": sigma_results_table,
                "from_nominal_cache": True,
            }
            continue

        point_totals = batched_control_totals[key]
        regions_table, estimators_table, estimator_map = compute_estimator_tables(
            lepton=lepton,
            data_counts=point_totals["data_counts"],
            mc_counts=point_totals["mc_counts"],
            clip_negative=bool(SETTINGS["ADDITIONAL_DATA_DRIVEN_BKG"]["CLIP_NEGATIVE_TO_ZERO"]),
        )
        selected_os = select_plotdict(
            plot_os,
            mass_window=mass_window,
            ptcone_max=ptcone_max,
            etcone_max=etcone_max,
            require_both=require_both,
        )
        sigma_results_table = build_sigma_results_table(
            plot_os=plot_os,
            channel_config=channel_config,
            produced_event_count_fn=produced_event_count_fn,
            mass_window=mass_window,
            ptcone_max=ptcone_max,
            etcone_max=etcone_max,
            require_both=require_both,
            estimator_map=estimator_map,
            produced_cache=produced_cache,
            methods=resolved_methods,
            selected_os=selected_os,
        )
        evaluations[(ptcone_max, etcone_max)] = {
            "regions_table": regions_table,
            "estimators_table": estimators_table,
            "sigma_results_table": sigma_results_table,
            "from_nominal_cache": False,
        }

    return evaluations


def serialise_cut_point_result(cut_point_result: dict) -> dict:
    serialised = dict(cut_point_result)
    for key in ("regions_table", "estimators_table", "sigma_results_table", "comparison_sigma_if_applied", "os_cutflow", "ss_cutflow"):
        value = serialised.get(key)
        if value is None:
            continue
        serialised[key] = value.to_dict(orient="records")
    return serialised


def save_cut_point_report(lepton: str, cut_point_result: dict, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    debug_outputs = save_control_region_debug_outputs(lepton, cut_point_result["stage_totals"], output_dir)
    regions_table = cut_point_result["regions_table"]
    estimators_table = cut_point_result["estimators_table"]
    sigma_results_table = cut_point_result["sigma_results_table"]
    comparison_sigma = cut_point_result["comparison_sigma_if_applied"]

    if SETTINGS["ADDITIONAL_DATA_DRIVEN_BKG"]["SAVE_DEBUG_TABLES"]:
        regions_table.to_csv(output_dir / f"{lepton}_regions_table.csv", index=False)
        estimators_table.to_csv(output_dir / f"{lepton}_estimators_table.csv", index=False)
        sigma_results_table.to_csv(output_dir / f"{lepton}_sigma_results_table.csv", index=False)
        comparison_sigma.to_csv(output_dir / f"{lepton}_comparison_sigma_if_applied.csv", index=False)

    if SETTINGS["ADDITIONAL_DATA_DRIVEN_BKG"]["SAVE_PLOTS"]:
        save_additional_bkg_plot(
            regions_table,
            output_dir / f"{lepton}_additional_data_driven_bkg_regions.png",
            title=f"{lepton}: control-region residuals with final cuts",
        )
        save_estimator_plot(
            estimators_table,
            output_dir / f"{lepton}_additional_data_driven_bkg_estimators.png",
            title=f"{lepton}: additional background estimators",
        )

    summary_lines = [
        f"Channel: {lepton}",
        "Cut consistency note: the same mass and isolation cuts are used for the signal region and the control-region DD estimate.",
        "",
        "Signal-region yields:",
        pd.DataFrame([cut_point_result["signal_region_yields"]]).to_string(index=False),
        "",
        "Regions table:",
        regions_table.to_string(index=False),
        "",
        "Estimators table:",
        estimators_table.to_string(index=False),
        "",
        "Sigma results table:",
        sigma_results_table.to_string(index=False),
        "",
        "Comparison sigma if applied:",
        comparison_sigma.to_string(index=False),
        "",
        "Control-region debug summary:",
        debug_outputs["summary"],
    ]
    write_text(output_dir / f"{lepton}_additional_data_driven_bkg_summary.txt", "\n".join(summary_lines))

    serialised = serialise_cut_point_result(cut_point_result)
    serialised["control_region_debug_summary"] = debug_outputs["summary"]
    if SETTINGS["ADDITIONAL_DATA_DRIVEN_BKG"]["SAVE_JSON"] and SETTINGS["SAVE_JSON"]:
        write_json(output_dir / f"{lepton}_cut_point_evaluation.json", serialised)

    return {
        "debug_summary": debug_outputs["summary"],
        "regions_table": regions_table,
        "estimators_table": estimators_table,
        "sigma_results_table": sigma_results_table,
        "comparison_sigma_if_applied": comparison_sigma,
    }
