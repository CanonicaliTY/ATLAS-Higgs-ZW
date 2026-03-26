from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from config import CHANNELS, SETTINGS, scan_dd_methods_for
from control_regions import evaluate_cut_point, evaluate_scan_cut_points, save_cut_point_report
from parquet_io import load_main_events
from scan import (
    build_local_stability_diagnostics,
    build_monotonicity_diagnostics,
    build_plateau_diagnostics,
    build_scan_diagnostics_table,
    local_stability_summary_text,
    monotonicity_summary_text,
    scan_isolation,
)
from systematics import evaluate_systematics
from utils import ensure_environment, ensure_script_directory, import_backend, log_step, now_stamp, progress_iter, write_json, write_text
from visualisation import (
    build_plot_dict,
    compose_image_grid,
    make_after_plots,
    make_before_plots,
    save_scan_heatmap,
    save_slice_plot,
    save_surface_plot,
    select_plotdict,
)


def _save_scan_diagnostics_outputs(
    lepton: str,
    diagnostic_table: pd.DataFrame,
    nominal_ptcone: float,
    nominal_etcone: float,
    scan_methods: tuple[str, ...],
    output_dir: Path,
) -> str:
    diagnostic_table.to_csv(output_dir / f"{lepton}_scan_diagnostics_full.csv", index=False)
    write_json(output_dir / f"{lepton}_scan_diagnostics_full.json", diagnostic_table.to_dict(orient="records"))

    plot_tasks: list[tuple[str, Callable[[], None]]] = []

    if SETTINGS["ISO_SCAN"]["SAVE_SIGMA_HEATMAPS"]:
        for method in scan_methods:
            plot_tasks.append(
                (
                    f"sigma heatmap ({method})",
                    lambda method=method: save_scan_heatmap(
                        diagnostic_table,
                        output_dir / f"{lepton}_heatmap_sigma_{method}.png",
                        value_column=f"sigma_pb_{method}",
                        value_label="sigma_pb [pb]",
                        title=f"{lepton}: extracted cross section ({method})",
                        nominal_ptcone=nominal_ptcone,
                        nominal_etcone=nominal_etcone,
                    ),
                )
            )

    if SETTINGS["ISO_SCAN"]["SAVE_SIGMA_3D_PLOTS"]:
        for method in scan_methods:
            plot_tasks.append(
                (
                    f"sigma surface ({method})",
                    lambda method=method: save_surface_plot(
                        diagnostic_table,
                        output_dir / f"{lepton}_surface_sigma_{method}.png",
                        value_column=f"sigma_pb_{method}",
                        value_label="sigma_pb [pb]",
                        title=f"{lepton}: sigma surface ({method})",
                        nominal_ptcone=nominal_ptcone,
                        nominal_etcone=nominal_etcone,
                    ),
                )
            )

    if SETTINGS["ISO_SCAN"]["SAVE_SIGMA_SLICE_PLOTS"]:
        for method in scan_methods:
            plot_tasks.append(
                (
                    f"sigma slice vs ptcone ({method})",
                    lambda method=method: save_slice_plot(
                        diagnostic_table,
                        output_dir / f"{lepton}_slice_sigma_{method}_vs_ptcone.png",
                        x_column="ptcone_max",
                        value_column=f"sigma_pb_{method}",
                        value_label="sigma_pb [pb]",
                        fixed_column="etcone_max",
                        fixed_value=nominal_etcone,
                        title=f"{lepton}: sigma vs ptcone through nominal etcone ({method})",
                    ),
                )
            )
            plot_tasks.append(
                (
                    f"sigma slice vs etcone ({method})",
                    lambda method=method: save_slice_plot(
                        diagnostic_table,
                        output_dir / f"{lepton}_slice_sigma_{method}_vs_etcone.png",
                        x_column="etcone_max",
                        value_column=f"sigma_pb_{method}",
                        value_label="sigma_pb [pb]",
                        fixed_column="ptcone_max",
                        fixed_value=nominal_ptcone,
                        title=f"{lepton}: sigma vs etcone through nominal ptcone ({method})",
                    ),
                )
            )

    if SETTINGS["ISO_SCAN"]["SAVE_SHIFT_HEATMAPS"]:
        for method in scan_methods:
            plot_tasks.append(
                (
                    f"absolute shift heatmap ({method})",
                    lambda method=method: save_scan_heatmap(
                        diagnostic_table,
                        output_dir / f"{lepton}_heatmap_abs_shift_{method}.png",
                        value_column=f"sigma_abs_shift_pb_{method}",
                        value_label="|Δsigma| [pb]",
                        title=f"{lepton}: |sigma(point) - sigma(nominal)| ({method})",
                        nominal_ptcone=nominal_ptcone,
                        nominal_etcone=nominal_etcone,
                    ),
                )
            )

    if SETTINGS["ISO_SCAN"]["SAVE_SHIFT_3D_PLOTS"]:
        for method in scan_methods:
            plot_tasks.append(
                (
                    f"absolute shift surface ({method})",
                    lambda method=method: save_surface_plot(
                        diagnostic_table,
                        output_dir / f"{lepton}_surface_abs_shift_{method}.png",
                        value_column=f"sigma_abs_shift_pb_{method}",
                        value_label="|Δsigma| [pb]",
                        title=f"{lepton}: |Δsigma| surface ({method})",
                        nominal_ptcone=nominal_ptcone,
                        nominal_etcone=nominal_etcone,
                    ),
                )
            )

    if SETTINGS["ISO_SCAN"]["SAVE_SHIFT_SLICE_PLOTS"]:
        for method in scan_methods:
            plot_tasks.append(
                (
                    f"absolute shift slice vs ptcone ({method})",
                    lambda method=method: save_slice_plot(
                        diagnostic_table,
                        output_dir / f"{lepton}_slice_abs_shift_{method}_vs_ptcone.png",
                        x_column="ptcone_max",
                        value_column=f"sigma_abs_shift_pb_{method}",
                        value_label="|Δsigma| [pb]",
                        fixed_column="etcone_max",
                        fixed_value=nominal_etcone,
                        title=f"{lepton}: |Δsigma| vs ptcone through nominal etcone ({method})",
                    ),
                )
            )
            plot_tasks.append(
                (
                    f"absolute shift slice vs etcone ({method})",
                    lambda method=method: save_slice_plot(
                        diagnostic_table,
                        output_dir / f"{lepton}_slice_abs_shift_{method}_vs_etcone.png",
                        x_column="etcone_max",
                        value_column=f"sigma_abs_shift_pb_{method}",
                        value_label="|Δsigma| [pb]",
                        fixed_column="ptcone_max",
                        fixed_value=nominal_ptcone,
                        title=f"{lepton}: |Δsigma| vs etcone through nominal ptcone ({method})",
                    ),
                )
            )

    if SETTINGS["ISO_SCAN"]["SAVE_FRACTIONAL_SHIFT_HEATMAPS"]:
        for method in scan_methods:
            plot_tasks.append(
                (
                    f"fractional shift heatmap ({method})",
                    lambda method=method: save_scan_heatmap(
                        diagnostic_table,
                        output_dir / f"{lepton}_heatmap_frac_shift_{method}.png",
                        value_column=f"sigma_frac_shift_{method}",
                        value_label="fractional |Δsigma|",
                        title=f"{lepton}: fractional |Δsigma| ({method})",
                        nominal_ptcone=nominal_ptcone,
                        nominal_etcone=nominal_etcone,
                    ),
                )
            )

    if SETTINGS["ISO_SCAN"]["SAVE_FRACTIONAL_SHIFT_3D_PLOTS"]:
        for method in scan_methods:
            plot_tasks.append(
                (
                    f"fractional shift surface ({method})",
                    lambda method=method: save_surface_plot(
                        diagnostic_table,
                        output_dir / f"{lepton}_surface_frac_shift_{method}.png",
                        value_column=f"sigma_frac_shift_{method}",
                        value_label="fractional |Δsigma|",
                        title=f"{lepton}: fractional |Δsigma| surface ({method})",
                        nominal_ptcone=nominal_ptcone,
                        nominal_etcone=nominal_etcone,
                    ),
                )
            )

    if SETTINGS["ISO_SCAN"]["SAVE_FRACTIONAL_SHIFT_SLICE_PLOTS"]:
        for method in scan_methods:
            plot_tasks.append(
                (
                    f"fractional shift slice vs ptcone ({method})",
                    lambda method=method: save_slice_plot(
                        diagnostic_table,
                        output_dir / f"{lepton}_slice_frac_shift_{method}_vs_ptcone.png",
                        x_column="ptcone_max",
                        value_column=f"sigma_frac_shift_{method}",
                        value_label="fractional |Δsigma|",
                        fixed_column="etcone_max",
                        fixed_value=nominal_etcone,
                        title=f"{lepton}: fractional |Δsigma| vs ptcone through nominal etcone ({method})",
                    ),
                )
            )
            plot_tasks.append(
                (
                    f"fractional shift slice vs etcone ({method})",
                    lambda method=method: save_slice_plot(
                        diagnostic_table,
                        output_dir / f"{lepton}_slice_frac_shift_{method}_vs_etcone.png",
                        x_column="etcone_max",
                        value_column=f"sigma_frac_shift_{method}",
                        value_label="fractional |Δsigma|",
                        fixed_column="ptcone_max",
                        fixed_value=nominal_ptcone,
                        title=f"{lepton}: fractional |Δsigma| vs etcone through nominal ptcone ({method})",
                    ),
                )
            )

    if plot_tasks:
        log_step(f"[{lepton}] Saving scan diagnostic plots")
        iterator = progress_iter(plot_tasks, total=len(plot_tasks), desc=f"{lepton} scan plots", unit="plot")
        for _, task in iterator:
            task()

    monotonicity = build_monotonicity_diagnostics(
        diagnostic_table=diagnostic_table,
        tolerance=float(SETTINGS["ISO_SCAN"]["MONOTONICITY_TOL_ABS"]),
        methods=scan_methods,
    )
    plateau = build_plateau_diagnostics(
        delta_table=monotonicity["delta_table"],
        tolerance=float(SETTINGS["ISO_SCAN"]["MONOTONICITY_TOL_ABS"]),
        methods=scan_methods,
    )
    local_stability = build_local_stability_diagnostics(
        diagnostic_table=diagnostic_table,
        nominal_ptcone=nominal_ptcone,
        nominal_etcone=nominal_etcone,
        methods=scan_methods,
        max_neighbours=int(SETTINGS["ISO_SCAN"]["LOCAL_STABILITY_NEIGHBOURS"]),
    )

    monotonicity["delta_table"].to_csv(output_dir / f"{lepton}_monotonicity_deltas.csv", index=False)
    monotonicity["classification_table"].to_csv(output_dir / f"{lepton}_monotonicity_classification.csv", index=False)
    plateau.to_csv(output_dir / f"{lepton}_plateau_diagnostics.csv", index=False)
    local_stability.to_csv(output_dir / f"{lepton}_local_stability.csv", index=False)

    summary_lines = [
        "Isolation scan diagnostics quantify residual dependence around the fixed nominal isolation working point.",
        "The scan is diagnostic/systematic input only; it does not choose the final cut.",
        f"Scan mode: {diagnostic_table['scan_mode'].iloc[0] if not diagnostic_table.empty else SETTINGS['ISO_SCAN']['SCAN_MODE']}",
        f"Evaluated DD methods: {', '.join(scan_methods)}",
        f"Scan points with defined sigma values: {int(diagnostic_table['sigma_defined'].sum())}/{len(diagnostic_table)}",
        "",
        "Sigma ranges across the diagnostic region:",
    ]
    for method in scan_methods:
        sigma_values = diagnostic_table[f"sigma_pb_{method}"].dropna()
        abs_shift_values = diagnostic_table[f"sigma_abs_shift_pb_{method}"].dropna()
        frac_shift_values = diagnostic_table[f"sigma_frac_shift_{method}"].dropna()
        if sigma_values.empty:
            summary_lines.append(f"{method}: no valid sigma values on this scan region")
            continue
        abs_shift_range = (
            f"[{abs_shift_values.min():.6f}, {abs_shift_values.max():.6f}] pb"
            if not abs_shift_values.empty
            else "unavailable"
        )
        frac_shift_range = (
            f"[{frac_shift_values.min():.6f}, {frac_shift_values.max():.6f}]"
            if not frac_shift_values.empty
            else "unavailable"
        )
        summary_lines.append(
            (
                f"{method}: sigma in [{sigma_values.min():.6f}, {sigma_values.max():.6f}] pb, "
                f"|Δsigma| in {abs_shift_range}, "
                f"fractional |Δsigma| in {frac_shift_range}"
            )
        )
    summary_lines.extend(
        [
            "",
            "Monotonicity and plateau summary:",
            monotonicity_summary_text(
                monotonicity["classification_table"],
                plateau_table=plateau,
                methods=scan_methods,
            ),
            "",
            "Local stability near the nominal point:",
            local_stability_summary_text(local_stability, methods=scan_methods),
            "",
            "Interpretation note:",
            "A clear plateau is not required. Monotonic sigma behaviour across the scanned region is an acceptable diagnostic outcome.",
        ]
    )
    summary_text = "\n".join(summary_lines)
    write_text(output_dir / f"{lepton}_scan_diagnostics_summary.txt", summary_text)
    return summary_text


def _write_scan_disabled_note(lepton: str, output_dir: Path) -> str:
    note = (
        "Isolation scan diagnostics were not produced for this run.\n"
        "Reason: ISO_SCAN['RUN_SCAN_DIAGNOSTICS'] was False.\n"
        "The nominal analysis still uses FIXED_ISO for the final result."
    )
    write_text(output_dir / f"{lepton}_scan_disabled.txt", note)
    return note


def run_channel(lepton: str, backend: dict, output_dir: Path) -> None:
    channel = CHANNELS[lepton]
    output_dir.mkdir(parents=True, exist_ok=True)
    log_step(f"[{lepton}] Starting channel")

    plot_stacked_hist = backend["plot_stacked_hist"]
    produced_event_count = backend["produced_event_count"]

    data_os = load_main_events(lepton, "OS", backend)
    data_ss = load_main_events(lepton, "SS", backend)

    plot_os = build_plot_dict(data_os, lepton)
    plot_ss = build_plot_dict(data_ss, lepton)

    before_dir = output_dir / "plots_before"
    before_paths = make_before_plots(lepton, plot_os, plot_ss, plot_stacked_hist, before_dir)
    if SETTINGS["MAKE_GROUP_FIGURES"] and SETTINGS["SAVE_PLOTS"]:
        compose_image_grid(
            before_paths,
            output_dir / f"{lepton}_beforeplots_grid.png",
            nrows=2,
            ncols=2,
            titles=["OS leading pT", "SS leading pT", "OS mass", "SS mass"],
            figsize=(16, 12),
        )

    mass_window = tuple(SETTINGS["MASS_WINDOW"])
    require_both = bool(SETTINGS["REQUIRE_BOTH_ISO"])
    nominal_ptcone = float(SETTINGS["FIXED_ISO"]["ptcone_max"])
    nominal_etcone = float(SETTINGS["FIXED_ISO"]["etcone_max"])
    run_scan_diagnostics = bool(SETTINGS["ISO_SCAN"]["RUN_SCAN_DIAGNOSTICS"])
    scan_methods = scan_dd_methods_for(lepton)

    plot_os_after = select_plotdict(
        plot_os,
        mass_window=mass_window,
        ptcone_max=nominal_ptcone,
        etcone_max=nominal_etcone,
        require_both=require_both,
    )
    plot_ss_after = select_plotdict(
        plot_ss,
        mass_window=mass_window,
        ptcone_max=nominal_ptcone,
        etcone_max=nominal_etcone,
        require_both=require_both,
    )

    after_dir = output_dir / "plots_after"
    after_paths = make_after_plots(
        lepton,
        plot_os_after,
        plot_ss_after,
        mass_window,
        nominal_ptcone,
        nominal_etcone,
        plot_stacked_hist,
        after_dir,
    )
    if SETTINGS["MAKE_GROUP_FIGURES"] and SETTINGS["SAVE_PLOTS"]:
        compose_image_grid(
            after_paths,
            output_dir / f"{lepton}_aftermass_grid.png",
            nrows=1,
            ncols=2,
            titles=["OS mass after cuts", "SS mass after cuts"],
            figsize=(14, 6),
        )

    control_cache: dict = {}
    scan_control_cache: dict = {}
    produced_cache: dict[str, float] = {}
    iso_dir = output_dir / "iso_scan"
    iso_dir.mkdir(parents=True, exist_ok=True)

    cut_point_result = evaluate_cut_point(
        lepton=lepton,
        plot_os=plot_os,
        plot_ss=plot_ss,
        channel_config=channel,
        produced_event_count_fn=produced_event_count,
        backend=backend,
        mass_window=mass_window,
        ptcone_max=nominal_ptcone,
        etcone_max=nominal_etcone,
        require_both=require_both,
        control_cache=control_cache,
        produced_cache=produced_cache,
        include_cutflows=True,
    )
    dd_report = save_cut_point_report(lepton, cut_point_result, output_dir / "additional_data_driven_bkg")

    diagnostic_table = None
    scan_diagnostic_summary = None
    if run_scan_diagnostics:
        log_step(f"[{lepton}] Running isolation scan diagnostics")
        scan_table = scan_isolation(
            plot_os=plot_os,
            plot_ss=plot_ss,
            mass_window=mass_window,
            require_both=require_both,
            nominal_ptcone=nominal_ptcone,
            nominal_etcone=nominal_etcone,
            scan_mode=str(SETTINGS["ISO_SCAN"]["SCAN_MODE"]),
            ptcone_range=tuple(SETTINGS["ISO_SCAN"]["PTCONE_RANGE"]),
            ptcone_step=float(SETTINGS["ISO_SCAN"]["PTCONE_STEP"]),
            etcone_range=tuple(SETTINGS["ISO_SCAN"]["ETCONE_RANGE"]),
            etcone_step=float(SETTINGS["ISO_SCAN"]["ETCONE_STEP"]),
            local_box_ptcone_half_width=float(SETTINGS["ISO_SCAN"]["LOCAL_BOX_PTCONE_HALF_WIDTH"]),
            local_box_etcone_half_width=float(SETTINGS["ISO_SCAN"]["LOCAL_BOX_ETCONE_HALF_WIDTH"]),
            progress_label=lepton,
        )

        scan_evaluations = evaluate_scan_cut_points(
            lepton=lepton,
            plot_os=plot_os,
            channel_config=channel,
            produced_event_count_fn=produced_event_count,
            backend=backend,
            mass_window=mass_window,
            cut_points=list(scan_table[["ptcone_max", "etcone_max"]].itertuples(index=False, name=None)),
            require_both=require_both,
            nominal_cut_point=cut_point_result,
            produced_cache=produced_cache,
            control_cache=scan_control_cache,
            methods=scan_methods,
        )
        evaluation_lookup = {
            (round(float(ptcone_max), 6), round(float(etcone_max), 6)): result
            for (ptcone_max, etcone_max), result in scan_evaluations.items()
        }

        def diagnostic_evaluator(ptcone_value: float, etcone_value: float) -> dict:
            return evaluation_lookup[(round(float(ptcone_value), 6), round(float(etcone_value), 6))]

        diagnostic_table = build_scan_diagnostics_table(
            scan_table=scan_table,
            cut_point_evaluator=diagnostic_evaluator,
            nominal_sigma_lookup=cut_point_result["sigma_results_table"].set_index("method"),
            methods=scan_methods,
        )
        scan_diagnostic_summary = _save_scan_diagnostics_outputs(
            lepton=lepton,
            diagnostic_table=diagnostic_table,
            nominal_ptcone=nominal_ptcone,
            nominal_etcone=nominal_etcone,
            scan_methods=scan_methods,
            output_dir=iso_dir,
        )
    else:
        scan_diagnostic_summary = _write_scan_disabled_note(lepton, iso_dir)

    systematics_result = evaluate_systematics(
        lepton=lepton,
        plot_os=plot_os,
        plot_ss=plot_ss,
        channel_config=channel,
        produced_event_count_fn=produced_event_count,
        backend=backend,
        nominal_cut_point=cut_point_result,
        mass_window=mass_window,
        ptcone_max=nominal_ptcone,
        etcone_max=nominal_etcone,
        require_both=require_both,
        systematics_dir=output_dir / "systematics",
        control_cache=control_cache,
        produced_cache=produced_cache,
        scan_diagnostic_table=diagnostic_table,
        scan_methods=scan_methods,
        scan_mode=str(SETTINGS["ISO_SCAN"]["SCAN_MODE"]) if run_scan_diagnostics else None,
    )

    uncertainty_table = pd.DataFrame(systematics_result["uncertainty_summary_by_method"])

    summary_lines = [
        f"Channel: {lepton}",
        "",
        "1. Nominal isolation working point",
        f"nominal_ptcone = {nominal_ptcone:.4f}",
        f"nominal_etcone = {nominal_etcone:.4f}",
        "This fixed nominal point is used for after-cut plots, cutflow, the main cross section result, and systematics.",
        "",
        "2. OS cutflow",
        cut_point_result["os_cutflow"].to_string(index=False),
        "",
        "3. SS cutflow",
        cut_point_result["ss_cutflow"].to_string(index=False) if cut_point_result["ss_cutflow"] is not None else "N/A",
        "",
        "4. Nominal cross section and uncertainty summary",
        uncertainty_table.to_string(index=False),
        "",
        "5. Additional-background comparison summary",
        dd_report["sigma_results_table"].to_string(index=False),
        "",
        "6. Regions table",
        dd_report["regions_table"].to_string(index=False),
        "",
        "7. Estimators table",
        dd_report["estimators_table"].to_string(index=False),
        "",
        "8. Comparison sigma if applied",
        dd_report["comparison_sigma_if_applied"].to_string(index=False),
        "",
        "9. Isolation scan diagnostic summary",
        scan_diagnostic_summary,
    ]

    write_text(output_dir / f"{lepton}_summary.txt", "\n".join(summary_lines))
    write_json(
        output_dir / f"{lepton}_cross_section.json",
        {
            "channel": lepton,
            "nominal_isolation_working_point": {
                "ptcone_max": nominal_ptcone,
                "etcone_max": nominal_etcone,
            },
            "mass_window": list(mass_window),
            "require_both_iso": require_both,
            "scan_diagnostics_mode": str(SETTINGS["ISO_SCAN"]["SCAN_MODE"]) if run_scan_diagnostics else None,
            "scan_dd_methods": list(scan_methods),
            "cut_point_evaluation": {
                "regions_table": dd_report["regions_table"].to_dict(orient="records"),
                "estimators_table": dd_report["estimators_table"].to_dict(orient="records"),
                "sigma_results_table": dd_report["sigma_results_table"].to_dict(orient="records"),
                "comparison_sigma_if_applied": dd_report["comparison_sigma_if_applied"].to_dict(orient="records"),
                "control_region_debug_summary": dd_report["debug_summary"],
            },
            "systematics": systematics_result,
            "scan_diagnostics_summary": scan_diagnostic_summary,
        },
    )
    log_step(f"[{lepton}] Finished channel")


def main() -> None:
    ensure_script_directory()
    ensure_environment()
    backend = import_backend()

    run_root = (Path(__file__).resolve().parent / SETTINGS["OUTPUT_DIR"] / f"run_{now_stamp()}").resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(run_root / "settings.json", SETTINGS)

    iterator = progress_iter(SETTINGS["LEPTONS"], total=len(SETTINGS["LEPTONS"]), desc="channels", unit="channel")
    for lepton in iterator:
        run_channel(lepton, backend, run_root / lepton)

    print(f"All done. Outputs saved under: {run_root}")


if __name__ == "__main__":
    main()
