from __future__ import annotations

from pathlib import Path

import awkward as ak
import pandas as pd

from config import (
    ALL_ESTIMATOR_METHODS,
    PHYSICAL_METHODS,
    SETTINGS,
    debug_method_names,
    nominal_order_mode,
)
from cross_section import compute_sigma
from parquet_io import accumulate_control_stage_totals_from_tight_chunks
from selections import apply_selection
from utils import write_json, write_text, yield_mc
from visualisation import save_additional_bkg_plot, save_estimator_plot


DEBUG_REGION_NAMES = ["ordered_11_13", "ordered_13_11", "ep_mum", "mup_em", "ee_ss", "mumu_ss"]
PHYSICAL_REGION_NAMES = ["ep_mum", "mup_em", "ee_ss", "mumu_ss"]
STAGE_LABELS = {
    "baseline": "baseline control preselection",
    "mass_only": "mass-only selection",
    "mass_plus_iso": "mass + iso selection",
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

    # The physical wrong-flavour regions are defined by flavour and charge only.
    # They must not depend on whether the ntuple stores the electron or muon first.
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
    counts = {region: float(len(events[masks[region]])) for region in DEBUG_REGION_NAMES}
    weights = {region: float(yield_mc(events[masks[region]])) for region in DEBUG_REGION_NAMES}
    return {"count": counts, "weight": weights}


def clip_value(value: float, clip_negative: bool) -> float:
    return max(float(value), 0.0) if clip_negative else float(value)


def wrong_charge_region_name(lepton: str) -> str:
    if lepton == "mu":
        return "mumu_ss"
    if lepton == "e":
        return "ee_ss"
    raise ValueError(f"Unsupported channel for wrong-charge estimator: {lepton!r}")


def stage_name_for_order_mode(order_mode: str) -> str:
    if order_mode == "recompute_after_iso":
        return "mass_plus_iso"
    if order_mode == "fixed_before_iso":
        return "mass_only"
    raise ValueError(f"Unsupported order mode: {order_mode!r}")


def _stage_selectors(
    mass_window: tuple[float, float],
    ptcone_max: float,
    etcone_max: float,
    require_both: bool,
):
    return {
        "baseline": lambda events: events,
        "mass_only": lambda events: apply_selection(events, mass_window=mass_window, require_both=require_both),
        "mass_plus_iso": lambda events: apply_selection(
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
    for stage_name in ("baseline", "mass_only", "mass_plus_iso"):
        stage_result = stage_totals[stage_name]
        ordered_11_13 = stage_result["data_counts"]["ordered_11_13"]
        ordered_13_11 = stage_result["data_counts"]["ordered_13_11"]
        physical_total = stage_result["data_counts"]["ep_mum"] + stage_result["data_counts"]["mup_em"]
        if physical_total > 0 and (ordered_11_13 == 0.0 or ordered_13_11 == 0.0):
            detected = True
            lines.append(
                f"{STAGE_LABELS[stage_name]}: ordered opposite-flavour counts are one-sided "
                f"((11,13)={ordered_11_13:.3f}, (13,11)={ordered_13_11:.3f}) while the physical "
                f"wrong-flavour regions remain populated. This is consistent with fixed ntuple lepton ordering."
            )
        else:
            lines.append(
                f"{STAGE_LABELS[stage_name]}: no strong one-sided ordered-pair population was seen "
                f"((11,13)={ordered_11_13:.3f}, (13,11)={ordered_13_11:.3f})."
            )

    if not detected:
        lines.insert(0, "No strong index-ordering issue was detected in the checked control-region stages.")
    else:
        lines.insert(0, "An index-ordering issue is present: ordered opposite-flavour counts are asymmetric even when physical wrong-flavour categories survive.")
    return "\n".join(lines)


def save_control_region_debug_outputs(lepton: str, stage_totals: dict, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    table = control_stage_totals_to_frame(stage_totals)
    summary_text = index_ordering_diagnostic(stage_totals)

    table.to_csv(output_dir / f"{lepton}_control_region_debug.csv", index=False)
    write_json(
        output_dir / f"{lepton}_control_region_debug.json",
        {
            "channel": lepton,
            "stage_totals": stage_totals,
            "summary": summary_text,
        },
    )
    write_text(output_dir / f"{lepton}_control_region_debug_summary.txt", summary_text + "\n\n" + table.to_string(index=False))

    return {"table": table, "summary": summary_text}


def compute_estimator_tables(
    lepton: str,
    data_counts: dict[str, float],
    mc_counts: dict[str, float],
    clip_negative: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    residuals = {region: data_counts[region] - mc_counts[region] for region in PHYSICAL_REGION_NAMES}
    clipped_residuals = {region: clip_value(value, clip_negative) for region, value in residuals.items()}

    region_rows = []
    for region in PHYSICAL_REGION_NAMES:
        region_rows.append(
            {
                "region": region,
                "N_data": data_counts[region],
                "N_MC": mc_counts[region],
                "residual": residuals[region],
                "clipped_residual": clipped_residuals[region],
            }
        )
    region_table = pd.DataFrame(region_rows)

    wrong_flavour_raw = 0.5 * (residuals["ep_mum"] + residuals["mup_em"])
    wrong_charge_region = wrong_charge_region_name(lepton)
    wrong_charge_raw = residuals[wrong_charge_region]
    both_average_raw = 0.5 * (wrong_flavour_raw + wrong_charge_raw)
    both_sum_raw = wrong_flavour_raw + wrong_charge_raw

    estimator_map = {
        "none": {"label": "No additional estimator applied", "raw": 0.0},
        "wrong_flavour": {"label": "Average wrong-flavour residual", "raw": wrong_flavour_raw},
        "wrong_charge": {"label": f"Wrong-charge residual from {wrong_charge_region}", "raw": wrong_charge_raw},
        "both_average": {"label": "Average of wrong-flavour and wrong-charge estimators", "raw": both_average_raw},
        "both_sum": {"label": "Sum of wrong-flavour and wrong-charge estimators (debug)", "raw": both_sum_raw},
    }
    for method in estimator_map:
        estimator_map[method]["clipped"] = clip_value(estimator_map[method]["raw"], clip_negative)

    estimator_rows = []
    for method in ALL_ESTIMATOR_METHODS:
        estimator_rows.append(
            {
                "method": method,
                "estimate_raw": estimator_map[method]["raw"],
                "estimate_clipped": estimator_map[method]["clipped"],
                "description": estimator_map[method]["label"],
                "included_in_total": method in PHYSICAL_METHODS,
            }
        )
    estimator_table = pd.DataFrame(estimator_rows)
    return region_table, estimator_table, estimator_map


def compute_estimator_detail(
    lepton: str,
    stage_totals: dict,
    method: str,
    order_mode: str,
    clip_negative: bool,
) -> dict:
    method = str(method).strip().lower()
    if method not in ALL_ESTIMATOR_METHODS:
        raise ValueError(f"Unknown additional-background method {method!r}")

    order_mode = nominal_order_mode() if order_mode == "compare_both" else order_mode
    stage_name = stage_name_for_order_mode(order_mode)
    stage_result = stage_totals[stage_name]
    data_counts = {region: stage_result["data_counts"][region] for region in PHYSICAL_REGION_NAMES}
    mc_counts = {region: stage_result["mc_counts"][region] for region in PHYSICAL_REGION_NAMES}
    region_table, estimator_table, estimator_map = compute_estimator_tables(
        lepton=lepton,
        data_counts=data_counts,
        mc_counts=mc_counts,
        clip_negative=clip_negative,
    )
    selected = estimator_map[method]
    return {
        "method": method,
        "order_mode": order_mode,
        "stage_name": stage_name,
        "stage_label": STAGE_LABELS[stage_name],
        "data_counts": data_counts,
        "mc_counts": mc_counts,
        "region_table": region_table,
        "estimator_table": estimator_table,
        "estimator_map": estimator_map,
        "selected_estimator_label": selected["label"],
        "extra_background_if_applied": float(selected["clipped"]),
    }


def evaluate_sigma_with_estimator(
    lepton: str,
    plot_os: dict,
    channel_config: dict,
    produced_event_count_fn,
    backend: dict,
    mass_window: tuple[float, float],
    ptcone_max: float,
    etcone_max: float,
    require_both: bool,
    method: str,
    order_mode: str,
    control_cache: dict | None = None,
    produced_cache: dict | None = None,
    stage_totals: dict | None = None,
) -> dict:
    sigma_without_additional_bkg = compute_sigma(
        plot_os=plot_os,
        channel_config=channel_config,
        produced_event_count_fn=produced_event_count_fn,
        mass_window=mass_window,
        ptcone_max=ptcone_max,
        etcone_max=etcone_max,
        require_both=require_both,
        extra_bkg=0.0,
        produced_sumw_cache=produced_cache,
    )

    if stage_totals is None:
        stage_totals = gather_control_stage_totals(
            backend=backend,
            lepton=lepton,
            mass_window=mass_window,
            ptcone_max=ptcone_max,
            etcone_max=etcone_max,
            require_both=require_both,
            cache=control_cache,
        )

    estimator_detail = compute_estimator_detail(
        lepton=lepton,
        stage_totals=stage_totals,
        method=method,
        order_mode=order_mode,
        clip_negative=bool(SETTINGS["ADDITIONAL_DATA_DRIVEN_BKG"]["CLIP_NEGATIVE_TO_ZERO"]),
    )

    extra_background_if_applied = (
        estimator_detail["extra_background_if_applied"]
        if SETTINGS["ADDITIONAL_DATA_DRIVEN_BKG"]["ENABLED"] and method != "none"
        else 0.0
    )
    sigma_if_applied = compute_sigma(
        plot_os=plot_os,
        channel_config=channel_config,
        produced_event_count_fn=produced_event_count_fn,
        mass_window=mass_window,
        ptcone_max=ptcone_max,
        etcone_max=etcone_max,
        require_both=require_both,
        extra_bkg=extra_background_if_applied,
        produced_sumw_cache=produced_cache,
    )

    return {
        "stage_totals": stage_totals,
        "estimator_detail": estimator_detail,
        "applied_extra_background": extra_background_if_applied,
        "sigma_without_additional_bkg": sigma_without_additional_bkg,
        "sigma_with_additional_bkg": sigma_if_applied,
    }


def build_method_comparison_table(
    lepton: str,
    plot_os: dict,
    channel_config: dict,
    produced_event_count_fn,
    mass_window: tuple[float, float],
    ptcone_max: float,
    etcone_max: float,
    require_both: bool,
    order_mode: str,
    stage_totals: dict,
    produced_cache: dict | None = None,
) -> pd.DataFrame:
    baseline_sigma = compute_sigma(
        plot_os=plot_os,
        channel_config=channel_config,
        produced_event_count_fn=produced_event_count_fn,
        mass_window=mass_window,
        ptcone_max=ptcone_max,
        etcone_max=etcone_max,
        require_both=require_both,
        extra_bkg=0.0,
        produced_sumw_cache=produced_cache,
    )
    rows = []
    for method in debug_method_names():
        detail = compute_estimator_detail(
            lepton=lepton,
            stage_totals=stage_totals,
            method=method,
            order_mode=order_mode,
            clip_negative=bool(SETTINGS["ADDITIONAL_DATA_DRIVEN_BKG"]["CLIP_NEGATIVE_TO_ZERO"]),
        )
        sigma_if_applied = compute_sigma(
            plot_os=plot_os,
            channel_config=channel_config,
            produced_event_count_fn=produced_event_count_fn,
            mass_window=mass_window,
            ptcone_max=ptcone_max,
            etcone_max=etcone_max,
            require_both=require_both,
            extra_bkg=detail["extra_background_if_applied"],
            produced_sumw_cache=produced_cache,
        )
        rows.append(
            {
                "method": method,
                "order_mode": order_mode,
                "stage_name": detail["stage_name"],
                "extra_bkg": detail["extra_background_if_applied"],
                "sigma_pb": sigma_if_applied["sigma_pb"],
                "sigma_shift_from_no_extra_pb": sigma_if_applied["sigma_pb"] - baseline_sigma["sigma_pb"],
                "included_in_total": method in PHYSICAL_METHODS,
            }
        )
    return pd.DataFrame(rows)


def build_order_comparison_table(
    lepton: str,
    plot_os: dict,
    channel_config: dict,
    produced_event_count_fn,
    mass_window: tuple[float, float],
    ptcone_max: float,
    etcone_max: float,
    require_both: bool,
    method: str,
    stage_totals: dict,
    produced_cache: dict | None = None,
) -> pd.DataFrame:
    rows = []
    for order_mode in ("recompute_after_iso", "fixed_before_iso"):
        detail = compute_estimator_detail(
            lepton=lepton,
            stage_totals=stage_totals,
            method=method,
            order_mode=order_mode,
            clip_negative=bool(SETTINGS["ADDITIONAL_DATA_DRIVEN_BKG"]["CLIP_NEGATIVE_TO_ZERO"]),
        )
        sigma_if_applied = compute_sigma(
            plot_os=plot_os,
            channel_config=channel_config,
            produced_event_count_fn=produced_event_count_fn,
            mass_window=mass_window,
            ptcone_max=ptcone_max,
            etcone_max=etcone_max,
            require_both=require_both,
            extra_bkg=detail["extra_background_if_applied"],
            produced_sumw_cache=produced_cache,
        )
        rows.append(
            {
                "order_mode": order_mode,
                "stage_name": detail["stage_name"],
                "extra_bkg": detail["extra_background_if_applied"],
                "sigma_pb": sigma_if_applied["sigma_pb"],
            }
        )
    comparison = pd.DataFrame(rows)
    if len(comparison) == 2:
        reference = comparison.loc[comparison["order_mode"] == "recompute_after_iso", "sigma_pb"].iloc[0]
        comparison["sigma_shift_from_recompute_pb"] = comparison["sigma_pb"] - reference
    return comparison


def build_additional_background_report(
    lepton: str,
    plot_os: dict,
    channel_config: dict,
    produced_event_count_fn,
    backend: dict,
    mass_window: tuple[float, float],
    ptcone_max: float,
    etcone_max: float,
    require_both: bool,
    output_dir: Path,
    control_cache: dict | None = None,
    produced_cache: dict | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_method = str(SETTINGS["ADDITIONAL_DATA_DRIVEN_BKG"]["METHOD"]).strip().lower()
    selected_order_mode = nominal_order_mode()

    selected_result = evaluate_sigma_with_estimator(
        lepton=lepton,
        plot_os=plot_os,
        channel_config=channel_config,
        produced_event_count_fn=produced_event_count_fn,
        backend=backend,
        mass_window=mass_window,
        ptcone_max=ptcone_max,
        etcone_max=etcone_max,
        require_both=require_both,
        method=selected_method,
        order_mode=selected_order_mode,
        control_cache=control_cache,
        produced_cache=produced_cache,
    )

    debug_outputs = save_control_region_debug_outputs(lepton, selected_result["stage_totals"], output_dir)
    estimator_detail = selected_result["estimator_detail"]
    method_comparison = build_method_comparison_table(
        lepton=lepton,
        plot_os=plot_os,
        channel_config=channel_config,
        produced_event_count_fn=produced_event_count_fn,
        mass_window=mass_window,
        ptcone_max=ptcone_max,
        etcone_max=etcone_max,
        require_both=require_both,
        order_mode=selected_order_mode,
        stage_totals=selected_result["stage_totals"],
        produced_cache=produced_cache,
    )
    order_comparison = build_order_comparison_table(
        lepton=lepton,
        plot_os=plot_os,
        channel_config=channel_config,
        produced_event_count_fn=produced_event_count_fn,
        mass_window=mass_window,
        ptcone_max=ptcone_max,
        etcone_max=etcone_max,
        require_both=require_both,
        method=selected_method,
        stage_totals=selected_result["stage_totals"],
        produced_cache=produced_cache,
    )

    region_table = estimator_detail["region_table"]
    estimator_table = estimator_detail["estimator_table"]

    if SETTINGS["ADDITIONAL_DATA_DRIVEN_BKG"]["SAVE_TABLES"]:
        region_table.to_csv(output_dir / f"{lepton}_additional_data_driven_bkg_regions.csv", index=False)
        estimator_table.to_csv(output_dir / f"{lepton}_additional_data_driven_bkg_estimators.csv", index=False)
        method_comparison.to_csv(output_dir / f"{lepton}_method_comparison.csv", index=False)
        order_comparison.to_csv(output_dir / f"{lepton}_order_mode_comparison.csv", index=False)

    if SETTINGS["ADDITIONAL_DATA_DRIVEN_BKG"]["SAVE_PLOTS"]:
        save_additional_bkg_plot(
            region_table,
            output_dir / f"{lepton}_additional_data_driven_bkg_regions.png",
            title=f"{lepton}: control-region residuals ({estimator_detail['stage_label']})",
        )
        save_estimator_plot(
            estimator_table,
            output_dir / f"{lepton}_additional_data_driven_bkg_estimators.png",
            title=f"{lepton}: additional background estimators",
            selected_method=selected_method,
        )

    summary_lines = [
        f"Channel: {lepton}",
        f"Selected method: {selected_method}",
        f"Selected order mode: {selected_order_mode}",
        f"Estimator stage: {estimator_detail['stage_label']}",
        (
            "Order-mode note: the estimator is cut-dependent. "
            "'recompute_after_iso' recomputes the control-region estimate after the final isolation cut, "
            "whereas 'fixed_before_iso' keeps the pre-isolation estimate fixed during the isolation scan."
        ),
        "",
        "Control-region debug summary:",
        debug_outputs["summary"],
        "",
        "Selected-stage region table:",
        region_table.to_string(index=False),
        "",
        "Estimator table:",
        estimator_table.to_string(index=False),
        "",
        "Method comparison:",
        method_comparison.to_string(index=False),
        "",
        "Order-mode comparison:",
        order_comparison.to_string(index=False),
        "",
        f"Sigma without additional background [pb]: {selected_result['sigma_without_additional_bkg']['sigma_pb']:.6f}",
        f"Sigma with selected estimator [pb]: {selected_result['sigma_with_additional_bkg']['sigma_pb']:.6f}",
        f"Applied additional background events: {selected_result['applied_extra_background']:.6f}",
    ]
    write_text(output_dir / f"{lepton}_additional_data_driven_bkg_summary.txt", "\n".join(summary_lines))

    result = {
        "channel": lepton,
        "selected_method": selected_method,
        "selected_order_mode": selected_order_mode,
        "mass_window": list(mass_window),
        "ptcone_max": ptcone_max,
        "etcone_max": etcone_max,
        "require_both_iso": require_both,
        "control_region_debug_summary": debug_outputs["summary"],
        "control_stage_totals": selected_result["stage_totals"],
        "selected_estimator": {
            "label": estimator_detail["selected_estimator_label"],
            "stage_name": estimator_detail["stage_name"],
            "stage_label": estimator_detail["stage_label"],
            "extra_background_if_applied": estimator_detail["extra_background_if_applied"],
            "region_table": region_table.to_dict(orient="records"),
            "estimator_table": estimator_table.to_dict(orient="records"),
        },
        "sigma_without_additional_bkg": selected_result["sigma_without_additional_bkg"],
        "sigma_with_additional_bkg": selected_result["sigma_with_additional_bkg"],
        "applied_extra_background": selected_result["applied_extra_background"],
        "method_comparison": method_comparison.to_dict(orient="records"),
        "order_mode_comparison": order_comparison.to_dict(orient="records"),
    }
    if SETTINGS["ADDITIONAL_DATA_DRIVEN_BKG"]["SAVE_JSON"] and SETTINGS["SAVE_JSON"]:
        write_json(output_dir / f"{lepton}_additional_data_driven_bkg.json", result)
    return result

