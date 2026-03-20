from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import DD_METHODS, SETTINGS
from control_regions import evaluate_cut_point, serialise_cut_point_result
from utils import log_step, progress_iter, write_json, write_text


def _mass_window_label(mass_window: tuple[float, float]) -> str:
    return f"{mass_window[0]:.1f}_{mass_window[1]:.1f}"


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
    produced_cache: dict | None = None,
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

    method_uncertainty_rows = []
    for method in DD_METHODS:
        method_rows = mass_variation_table[mass_variation_table["method"] == method]
        dsigma_mass_window_pb = (
            float(method_rows["sigma_shift_pb"].abs().max()) if not method_rows.empty else 0.0
        )
        method_uncertainty_rows.append(
            {
                "method": method,
                "sigma_pb": float(nominal_sigma_lookup.loc[method, "sigma_pb"]),
                "dsigma_stat_pb": float(nominal_sigma_lookup.loc[method, "dsigma_stat_pb"]),
                "dsigma_lumi_pb": float(nominal_sigma_lookup.loc[method, "dsigma_lumi_pb"]),
                "dsigma_mass_window_pb": dsigma_mass_window_pb,
            }
        )
    method_uncertainty_table = pd.DataFrame(method_uncertainty_rows)
    method_uncertainty_table.to_csv(systematics_dir / f"{lepton}_uncertainty_summary_by_method.csv", index=False)

    summary_lines = [
        f"Channel: {lepton}",
        "Official scalar isolation systematic is not defined automatically here.",
        "Use the scan diagnostics and monotonicity summaries to judge the isolation dependence.",
        "",
        "Per-method nominal values and uncertainties:",
        method_uncertainty_table.to_string(index=False),
        "",
        "Mass-window variations:",
        mass_variation_table.to_string(index=False),
    ]
    write_text(systematics_dir / f"{lepton}_systematic_summary.txt", "\n".join(summary_lines))

    result = {
        "channel": lepton,
        "nominal_cut_point": serialise_cut_point_result(nominal_cut_point),
        "mass_window": {
            "variation_table": mass_variation_table.to_dict(orient="records"),
            "uncertainty_summary_by_method": method_uncertainty_table.to_dict(orient="records"),
            "detailed_variations": detailed_variations,
        },
        "isolation_systematic": None,
        "notes": [
            "No order-mode systematic is defined.",
            "No neighbour-difference isolation systematic is defined.",
            "Isolation dependence should be inspected through the scan diagnostics outputs.",
        ],
    }
    write_json(systematics_dir / f"{lepton}_systematics.json", result)
    return result
