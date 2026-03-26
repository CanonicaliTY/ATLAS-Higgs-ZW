from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from config import DD_METHODS, SETTINGS
from control_regions import evaluate_cut_point, serialise_cut_point_result
from utils import log_step, progress_iter, write_json, write_text


def _mass_window_label(mass_window: tuple[float, float]) -> str:
    return f"{mass_window[0]:.1f}_{mass_window[1]:.1f}"


def _resolved_methods(methods: Sequence[str] | None) -> tuple[str, ...]:
    if methods is None:
        return DD_METHODS
    ordered: list[str] = []
    for method in methods:
        if method not in DD_METHODS:
            raise ValueError(f"Unsupported DD method {method!r}")
        if method not in ordered:
            ordered.append(method)
    return tuple(ordered)


def _build_isolation_dependence_table(
    nominal_sigma_lookup: pd.DataFrame,
    scan_diagnostic_table: pd.DataFrame | None,
    methods: Sequence[str],
    scan_mode: str | None,
) -> pd.DataFrame:
    rows = []
    if scan_diagnostic_table is None or scan_diagnostic_table.empty:
        return pd.DataFrame(rows)

    for method in methods:
        abs_shift_column = f"sigma_abs_shift_pb_{method}"
        frac_shift_column = f"sigma_frac_shift_{method}"
        sigma_column = f"sigma_pb_{method}"
        if abs_shift_column not in scan_diagnostic_table.columns:
            continue

        method_rows = scan_diagnostic_table.dropna(subset=[abs_shift_column]).copy()
        if method_rows.empty:
            rows.append(
                {
                    "method": method,
                    "sigma_nominal_pb": float(nominal_sigma_lookup.loc[method, "sigma_pb"]),
                    "dsigma_isolation_pb": 0.0,
                    "max_fractional_shift": 0.0,
                    "sigma_min_pb": float("nan"),
                    "sigma_max_pb": float("nan"),
                    "max_shift_ptcone": float("nan"),
                    "max_shift_etcone": float("nan"),
                    "scan_mode": scan_mode,
                    "n_scan_points": 0,
                }
            )
            continue

        max_index = method_rows[abs_shift_column].idxmax()
        max_row = method_rows.loc[max_index]
        rows.append(
            {
                "method": method,
                "sigma_nominal_pb": float(nominal_sigma_lookup.loc[method, "sigma_pb"]),
                "dsigma_isolation_pb": float(method_rows[abs_shift_column].max()),
                "max_fractional_shift": float(method_rows[frac_shift_column].max()),
                "sigma_min_pb": float(method_rows[sigma_column].min()),
                "sigma_max_pb": float(method_rows[sigma_column].max()),
                "max_shift_ptcone": float(max_row["ptcone_max"]),
                "max_shift_etcone": float(max_row["etcone_max"]),
                "scan_mode": scan_mode,
                "n_scan_points": int(len(method_rows)),
            }
        )
    return pd.DataFrame(rows)


def evaluate_systematics(
    lepton: str,
    plot_os: dict,
    plot_ss: dict | None,
    channel_config: dict,
    produced_event_count_fn,
    backend: dict,
    nominal_cut_point: dict,
    mass_window: tuple[float, float],
    ptcone_max: float,
    etcone_max: float,
    require_both: bool,
    systematics_dir: Path,
    control_cache: dict | None = None,
    produced_cache: dict[str, float] | None = None,
    scan_diagnostic_table: pd.DataFrame | None = None,
    scan_methods: Sequence[str] | None = None,
    scan_mode: str | None = None,
) -> dict:
    systematics_dir.mkdir(parents=True, exist_ok=True)

    nominal_sigma_table = nominal_cut_point["sigma_results_table"].copy()
    nominal_sigma_lookup = nominal_sigma_table.set_index("method")

    mass_variation_rows = []
    detailed_variations = []
    variations = SETTINGS["SYSTEMATICS"]["MASS_WINDOW_VARIATIONS"]
    log_step(f"[{lepton}] Evaluating mass-window variations")
    iterator = progress_iter(variations, total=len(variations), desc=f"{lepton} mass syst", unit="window")
    for varied_mass_window in iterator:
        if float(varied_mass_window[0]) <= 40.0:
            raise ValueError(
                "Mass-window variations must stay above the 40 GeV threshold of the primary signal samples. "
                f"Got {varied_mass_window}."
            )

        varied_cut_point = evaluate_cut_point(
            lepton=lepton,
            plot_os=plot_os,
            plot_ss=plot_ss,
            channel_config=channel_config,
            produced_event_count_fn=produced_event_count_fn,
            backend=backend,
            mass_window=tuple(varied_mass_window),
            ptcone_max=ptcone_max,
            etcone_max=etcone_max,
            require_both=require_both,
            control_cache=control_cache,
            produced_cache=produced_cache,
            include_cutflows=False,
        )
        detailed_variations.append(
            {
                "variation": _mass_window_label(tuple(varied_mass_window)),
                "mass_window": list(varied_mass_window),
                "cut_point": serialise_cut_point_result(varied_cut_point),
            }
        )

        varied_sigma_lookup = varied_cut_point["sigma_results_table"].set_index("method")
        for method in DD_METHODS:
            nominal_sigma = float(nominal_sigma_lookup.loc[method, "sigma_pb"])
            varied_sigma = float(varied_sigma_lookup.loc[method, "sigma_pb"])
            mass_variation_rows.append(
                {
                    "variation": _mass_window_label(tuple(varied_mass_window)),
                    "mass_low": float(varied_mass_window[0]),
                    "mass_high": float(varied_mass_window[1]),
                    "method": method,
                    "sigma_pb": varied_sigma,
                    "sigma_shift_pb": varied_sigma - nominal_sigma,
                    "extra_bkg": float(varied_sigma_lookup.loc[method, "extra_bkg"]),
                }
            )

    mass_variation_table = pd.DataFrame(mass_variation_rows)
    mass_variation_table.to_csv(systematics_dir / f"{lepton}_mass_window_variations.csv", index=False)

    resolved_scan_methods = _resolved_methods(scan_methods)
    isolation_dependence_table = _build_isolation_dependence_table(
        nominal_sigma_lookup=nominal_sigma_lookup,
        scan_diagnostic_table=scan_diagnostic_table,
        methods=resolved_scan_methods,
        scan_mode=scan_mode,
    )
    isolation_dependence_table.to_csv(systematics_dir / f"{lepton}_isolation_dependence.csv", index=False)
    isolation_lookup = (
        isolation_dependence_table.set_index("method")
        if not isolation_dependence_table.empty
        else pd.DataFrame(columns=["dsigma_isolation_pb"])
    )

    method_uncertainty_rows = []
    for method in DD_METHODS:
        method_rows = mass_variation_table[mass_variation_table["method"] == method]
        dsigma_mass_window_pb = (
            float(method_rows["sigma_shift_pb"].abs().max()) if not method_rows.empty else 0.0
        )
        dsigma_isolation_pb = (
            float(isolation_lookup.loc[method, "dsigma_isolation_pb"])
            if not isolation_lookup.empty and method in isolation_lookup.index
            else 0.0
        )
        method_uncertainty_rows.append(
            {
                "method": method,
                "sigma_pb": float(nominal_sigma_lookup.loc[method, "sigma_pb"]),
                "dsigma_stat_pb": float(nominal_sigma_lookup.loc[method, "dsigma_stat_pb"]),
                "dsigma_lumi_pb": float(nominal_sigma_lookup.loc[method, "dsigma_lumi_pb"]),
                "dsigma_mass_window_pb": dsigma_mass_window_pb,
                "dsigma_isolation_pb": dsigma_isolation_pb,
            }
        )
    method_uncertainty_table = pd.DataFrame(method_uncertainty_rows)
    method_uncertainty_table.to_csv(systematics_dir / f"{lepton}_uncertainty_summary_by_method.csv", index=False)

    isolation_summary_text = (
        isolation_dependence_table.to_string(index=False)
        if not isolation_dependence_table.empty
        else "Isolation-dependence summary unavailable because scan diagnostics were not run."
    )
    summary_lines = [
        f"Channel: {lepton}",
        "Isolation dependence is derived from scan diagnostics relative to the fixed nominal isolation working point.",
        "This is a dependence study over the configured scan region, not a best-point optimiser.",
        "",
        "Per-method nominal values and uncertainties:",
        method_uncertainty_table.to_string(index=False),
        "",
        "Mass-window variations:",
        mass_variation_table.to_string(index=False),
        "",
        "Isolation dependence summary:",
        isolation_summary_text,
    ]
    write_text(systematics_dir / f"{lepton}_systematic_summary.txt", "\n".join(summary_lines))

    isolation_dependence_result = {
        "scan_mode": scan_mode,
        "summary_by_method": isolation_dependence_table.to_dict(orient="records"),
        "notes": [
            "dsigma_isolation_pb is the maximum absolute sigma shift relative to the nominal FIXED_ISO point over the configured diagnostic scan region.",
            "The isolation scan is used to quantify residual dependence, not to select an optimal cut.",
        ],
    }

    result = {
        "channel": lepton,
        "nominal_cut_point": serialise_cut_point_result(nominal_cut_point),
        "mass_window": {
            "variation_table": mass_variation_table.to_dict(orient="records"),
            "detailed_variations": detailed_variations,
        },
        "isolation_dependence": isolation_dependence_result,
        "isolation_systematic": isolation_dependence_result,
        "uncertainty_summary_by_method": method_uncertainty_table.to_dict(orient="records"),
        "notes": [
            "No order-mode systematic is defined.",
            "Isolation dependence is reported relative to the fixed nominal working point.",
            "A plateau is not required for the scan to be usable as a dependence study.",
        ],
    }
    write_json(systematics_dir / f"{lepton}_systematics.json", result)
    return result
