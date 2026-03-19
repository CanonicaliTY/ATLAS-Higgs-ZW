from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from config import PHYSICAL_METHODS, SETTINGS, nominal_order_mode
from control_regions import (
    build_method_comparison_table,
    build_order_comparison_table,
    evaluate_sigma_with_estimator,
)
from scan import find_scan_neighbours, scan_table_with_sigma
from utils import write_json, write_text


def _mass_window_label(mass_window: tuple[float, float]) -> str:
    return f"{mass_window[0]:.1f}_{mass_window[1]:.1f}"


def combine_systematics(components: dict[str, float], mode: str) -> float:
    values = [abs(float(value)) for value in components.values()]
    if not values:
        return 0.0
    if mode == "conservative_envelope":
        return max(values)
    if mode == "quadrature":
        return math.sqrt(sum(value * value for value in values))
    raise ValueError(f"Unknown systematic combination mode: {mode!r}")


def evaluate_isolation_neighbour_systematic(
    sigma_neighbour_table: pd.DataFrame,
    nominal_sigma_pb: float,
    output_path: Path,
) -> dict:
    shifts = []
    for row in sigma_neighbour_table.itertuples(index=False):
        shifts.append(abs(float(row.sigma_pb) - nominal_sigma_pb))
    value_pb = max(shifts) if shifts else 0.0
    sigma_neighbour_table.to_csv(output_path, index=False)
    return {
        "nominal_sigma_pb": nominal_sigma_pb,
        "value_pb": value_pb,
        "neighbours": sigma_neighbour_table.to_dict(orient="records"),
    }


def evaluate_systematics(
    lepton: str,
    plot_os: dict,
    channel_config: dict,
    produced_event_count_fn,
    backend: dict,
    mass_window: tuple[float, float],
    ptcone_max: float,
    etcone_max: float,
    require_both: bool,
    full_scan: pd.DataFrame | None,
    allowed_scan: pd.DataFrame | None,
    systematics_dir: Path,
    scan_output_dir: Path | None = None,
    control_cache: dict | None = None,
    produced_cache: dict | None = None,
) -> dict:
    systematics_dir.mkdir(parents=True, exist_ok=True)

    nominal_method = str(SETTINGS["ADDITIONAL_DATA_DRIVEN_BKG"]["METHOD"]).strip().lower()
    selected_order_mode = nominal_order_mode()

    nominal_result = evaluate_sigma_with_estimator(
        lepton=lepton,
        plot_os=plot_os,
        channel_config=channel_config,
        produced_event_count_fn=produced_event_count_fn,
        backend=backend,
        mass_window=mass_window,
        ptcone_max=ptcone_max,
        etcone_max=etcone_max,
        require_both=require_both,
        method=nominal_method,
        order_mode=selected_order_mode,
        control_cache=control_cache,
        produced_cache=produced_cache,
    )
    nominal_sigma = nominal_result["sigma_with_additional_bkg"]
    nominal_sigma_pb = nominal_sigma["sigma_pb"]

    sigma_allowed_table = None
    isolation_neighbour = {
        "nominal_sigma_pb": nominal_sigma_pb,
        "value_pb": 0.0,
        "neighbours": [],
        "available": False,
    }
    isolation_global = {
        "nominal_sigma_pb": nominal_sigma_pb,
        "value_pb": 0.0,
        "available": False,
        "scan_points": [],
    }

    if full_scan is not None and allowed_scan is not None:
        def sigma_evaluator(scan_ptcone: float, scan_etcone: float) -> dict:
            return evaluate_sigma_with_estimator(
                lepton=lepton,
                plot_os=plot_os,
                channel_config=channel_config,
                produced_event_count_fn=produced_event_count_fn,
                backend=backend,
                mass_window=mass_window,
                ptcone_max=scan_ptcone,
                etcone_max=scan_etcone,
                require_both=require_both,
                method=nominal_method,
                order_mode=selected_order_mode,
                control_cache=control_cache,
                produced_cache=produced_cache,
            )

        sigma_allowed_table = scan_table_with_sigma(
            allowed_scan,
            sigma_evaluator=sigma_evaluator,
            method=nominal_method,
            order_mode=selected_order_mode,
        )
        sigma_allowed_table["sigma_shift_from_nominal_pb"] = sigma_allowed_table["sigma_pb"] - nominal_sigma_pb
        if scan_output_dir is not None:
            scan_output_dir.mkdir(parents=True, exist_ok=True)
            sigma_allowed_table.to_csv(scan_output_dir / f"{lepton}_sigma_scan_allowed_after_additional_bkg.csv", index=False)

        neighbour_points = find_scan_neighbours(
            full_scan,
            nominal_ptcone=ptcone_max,
            nominal_etcone=etcone_max,
            ptcone_step=float(SETTINGS["ISO_SCAN"]["PTCONE_STEP"]),
            etcone_step=float(SETTINGS["ISO_SCAN"]["ETCONE_STEP"]),
        )

        neighbour_rows = []
        for neighbour in neighbour_points.to_dict(orient="records"):
            evaluation = sigma_evaluator(float(neighbour["ptcone_max"]), float(neighbour["etcone_max"]))
            sigma_value = evaluation["sigma_with_additional_bkg"]
            neighbour_rows.append(
                {
                    "ptcone_max": float(neighbour["ptcone_max"]),
                    "etcone_max": float(neighbour["etcone_max"]),
                    "extra_bkg": evaluation["applied_extra_background"],
                    "sigma_pb": sigma_value["sigma_pb"],
                    "sigma_shift_from_nominal_pb": sigma_value["sigma_pb"] - nominal_sigma_pb,
                }
            )
        neighbour_table = pd.DataFrame(neighbour_rows)
        isolation_neighbour = evaluate_isolation_neighbour_systematic(
            neighbour_table,
            nominal_sigma_pb=nominal_sigma_pb,
            output_path=systematics_dir / f"{lepton}_isolation_neighbours.csv",
        )
        isolation_neighbour["available"] = True

        isolation_global = {
            "nominal_sigma_pb": nominal_sigma_pb,
            "value_pb": float(np.max(np.abs(sigma_allowed_table["sigma_shift_from_nominal_pb"]))) if not sigma_allowed_table.empty else 0.0,
            "available": True,
            "scan_points": sigma_allowed_table.to_dict(orient="records"),
        }
        if scan_output_dir is not None:
            write_text(
                scan_output_dir / f"{lepton}_sigma_scan_allowed_after_additional_bkg_summary.txt",
                (
                    f"Nominal sigma [pb]: {nominal_sigma_pb:.6f}\n"
                    f"Local neighbour systematic [pb]: {isolation_neighbour['value_pb']:.6f}\n"
                    f"Allowed-scan envelope [pb]: {isolation_global['value_pb']:.6f}\n"
                ),
            )

    mass_rows = []
    for varied_mass_window in SETTINGS["SYSTEMATICS"]["MASS_WINDOW_VARIATIONS"]:
        if float(varied_mass_window[0]) <= 40.0:
            raise ValueError(
                "Mass-window variations must stay above the 40 GeV threshold of the primary signal samples. "
                f"Got {varied_mass_window}."
            )
        evaluation = evaluate_sigma_with_estimator(
            lepton=lepton,
            plot_os=plot_os,
            channel_config=channel_config,
            produced_event_count_fn=produced_event_count_fn,
            backend=backend,
            mass_window=tuple(varied_mass_window),
            ptcone_max=ptcone_max,
            etcone_max=etcone_max,
            require_both=require_both,
            method=nominal_method,
            order_mode=selected_order_mode,
            control_cache=control_cache,
            produced_cache=produced_cache,
        )
        sigma_value = evaluation["sigma_with_additional_bkg"]
        mass_rows.append(
            {
                "variation": _mass_window_label(tuple(varied_mass_window)),
                "mass_low": float(varied_mass_window[0]),
                "mass_high": float(varied_mass_window[1]),
                "extra_bkg": evaluation["applied_extra_background"],
                "sigma_pb": sigma_value["sigma_pb"],
                "sigma_shift_from_nominal_pb": sigma_value["sigma_pb"] - nominal_sigma_pb,
            }
        )
    mass_table = pd.DataFrame(mass_rows)
    mass_table.to_csv(systematics_dir / f"{lepton}_mass_window_systematics.csv", index=False)
    mass_systematic_pb = float(np.max(np.abs(mass_table["sigma_shift_from_nominal_pb"]))) if not mass_table.empty else 0.0

    method_table = build_method_comparison_table(
        lepton=lepton,
        plot_os=plot_os,
        channel_config=channel_config,
        produced_event_count_fn=produced_event_count_fn,
        mass_window=mass_window,
        ptcone_max=ptcone_max,
        etcone_max=etcone_max,
        require_both=require_both,
        order_mode=selected_order_mode,
        stage_totals=nominal_result["stage_totals"],
        produced_cache=produced_cache,
    )
    method_table["sigma_shift_from_nominal_pb"] = method_table["sigma_pb"] - nominal_sigma_pb
    method_table.to_csv(systematics_dir / f"{lepton}_method_systematics.csv", index=False)
    included_methods = method_table[method_table["included_in_total"]]
    method_systematic_pb = (
        float(np.max(np.abs(included_methods["sigma_shift_from_nominal_pb"]))) if not included_methods.empty else 0.0
    )

    order_table = build_order_comparison_table(
        lepton=lepton,
        plot_os=plot_os,
        channel_config=channel_config,
        produced_event_count_fn=produced_event_count_fn,
        mass_window=mass_window,
        ptcone_max=ptcone_max,
        etcone_max=etcone_max,
        require_both=require_both,
        method=nominal_method,
        stage_totals=nominal_result["stage_totals"],
        produced_cache=produced_cache,
    )
    order_table["sigma_shift_from_nominal_pb"] = order_table["sigma_pb"] - nominal_sigma_pb
    order_table.to_csv(systematics_dir / f"{lepton}_order_systematics.csv", index=False)
    order_systematic_pb = (
        abs(float(order_table["sigma_pb"].max()) - float(order_table["sigma_pb"].min())) if not order_table.empty else 0.0
    )

    component_values = {
        "isolation_neighbour": isolation_neighbour["value_pb"],
        "mass_window": mass_systematic_pb,
        "estimator_method": method_systematic_pb,
        "estimator_order": order_systematic_pb,
    }
    total_systematic_pb = combine_systematics(component_values, SETTINGS["SYSTEMATICS"]["COMBINATION_MODE"])

    breakdown_rows = [
        {
            "component": "isolation_neighbour",
            "nominal_sigma_pb": nominal_sigma_pb,
            "value_pb": isolation_neighbour["value_pb"],
            "included_in_total": True,
            "note": "Local grid-neighbour envelope around the nominal isolation working point.",
        },
        {
            "component": "isolation_global_allowed_envelope",
            "nominal_sigma_pb": nominal_sigma_pb,
            "value_pb": isolation_global["value_pb"],
            "included_in_total": False,
            "note": "Diagnostic envelope over the allowed isolation scan.",
        },
        {
            "component": "mass_window",
            "nominal_sigma_pb": nominal_sigma_pb,
            "value_pb": mass_systematic_pb,
            "included_in_total": True,
            "note": "Small Z-window variations around the nominal mass window.",
        },
        {
            "component": "estimator_method",
            "nominal_sigma_pb": nominal_sigma_pb,
            "value_pb": method_systematic_pb,
            "included_in_total": True,
            "note": "Spread among wrong_flavour, wrong_charge, and both_average.",
        },
        {
            "component": "estimator_order",
            "nominal_sigma_pb": nominal_sigma_pb,
            "value_pb": order_systematic_pb,
            "included_in_total": True,
            "note": "Difference between recompute_after_iso and fixed_before_iso.",
        },
    ]
    breakdown_table = pd.DataFrame(breakdown_rows)
    breakdown_table.to_csv(systematics_dir / f"{lepton}_systematic_breakdown.csv", index=False)

    summary_lines = [
        f"Channel: {lepton}",
        f"Nominal method: {nominal_method}",
        f"Nominal order mode: {selected_order_mode}",
        f"Nominal sigma [pb]: {nominal_sigma_pb:.6f}",
        f"Systematic combination mode: {SETTINGS['SYSTEMATICS']['COMBINATION_MODE']}",
        f"Total systematic [pb]: {total_systematic_pb:.6f}",
        "",
        "Breakdown:",
        breakdown_table.to_string(index=False),
    ]
    write_text(systematics_dir / f"{lepton}_systematic_summary.txt", "\n".join(summary_lines))

    result = {
        "channel": lepton,
        "nominal_settings": {
            "method": nominal_method,
            "order_mode": selected_order_mode,
            "mass_window": list(mass_window),
            "ptcone_max": ptcone_max,
            "etcone_max": etcone_max,
            "require_both_iso": require_both,
        },
        "nominal_sigma": nominal_sigma,
        "components": {
            "isolation_neighbour": isolation_neighbour,
            "isolation_global_allowed_envelope": isolation_global,
            "mass_window": {
                "nominal_sigma_pb": nominal_sigma_pb,
                "value_pb": mass_systematic_pb,
                "variations": mass_table.to_dict(orient="records"),
            },
            "estimator_method": {
                "nominal_sigma_pb": nominal_sigma_pb,
                "value_pb": method_systematic_pb,
                "comparisons": method_table.to_dict(orient="records"),
            },
            "estimator_order": {
                "nominal_sigma_pb": nominal_sigma_pb,
                "value_pb": order_systematic_pb,
                "comparisons": order_table.to_dict(orient="records"),
            },
        },
        "total_systematic_pb": total_systematic_pb,
        "combination_mode": SETTINGS["SYSTEMATICS"]["COMBINATION_MODE"],
    }
    write_json(systematics_dir / f"{lepton}_systematics.json", result)
    return result

