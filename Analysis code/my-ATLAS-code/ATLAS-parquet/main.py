from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import CHANNELS, DD_METHODS, SETTINGS, iso_eff_threshold_for
from control_regions import evaluate_cut_point, save_cut_point_report
from parquet_io import load_main_events
from scan import build_monotonicity_diagnostics, build_scan_diagnostics_table, monotonicity_summary_text, scan_isolation
from systematics import evaluate_systematics
from utils import ensure_environment, ensure_script_directory, import_backend, now_stamp, write_json, write_text
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
    diagnostic_table,
    best_ptcone: float,
    best_etcone: float,
    output_dir: Path,
) -> str:
    diagnostic_table.to_csv(output_dir / f"{lepton}_scan_diagnostics_full.csv", index=False)
    write_json(output_dir / f"{lepton}_scan_diagnostics_full.json", diagnostic_table.to_dict(orient="records"))

    if SETTINGS["ISO_SCAN"]["SAVE_HEATMAPS"]:
        save_scan_heatmap(
            diagnostic_table,
            output_dir / f"{lepton}_heatmap_significance.png",
            value_column="significance",
            title=f"{lepton}: significance",
            best_ptcone=best_ptcone,
            best_etcone=best_etcone,
        )
        for method in DD_METHODS:
            save_scan_heatmap(
                diagnostic_table,
                output_dir / f"{lepton}_heatmap_sigma_{method}.png",
                value_column=f"sigma_pb_{method}",
                title=f"{lepton}: sigma ({method})",
                best_ptcone=best_ptcone,
                best_etcone=best_etcone,
            )

    if SETTINGS["ISO_SCAN"]["SAVE_3D_PLOTS"]:
        save_surface_plot(
            diagnostic_table,
            output_dir / f"{lepton}_surface_significance.png",
            value_column="significance",
            title=f"{lepton}: significance surface",
        )
        for method in DD_METHODS:
            save_surface_plot(
                diagnostic_table,
                output_dir / f"{lepton}_surface_sigma_{method}.png",
                value_column=f"sigma_pb_{method}",
                title=f"{lepton}: sigma surface ({method})",
            )

    if SETTINGS["ISO_SCAN"]["SAVE_SLICE_PLOTS"]:
        for method in DD_METHODS:
            save_slice_plot(
                diagnostic_table,
                output_dir / f"{lepton}_slice_sigma_{method}_vs_ptcone.png",
                x_column="ptcone_max",
                value_column=f"sigma_pb_{method}",
                fixed_column="etcone_max",
                title=f"{lepton}: sigma ({method}) vs ptcone at fixed etcone",
            )
            save_slice_plot(
                diagnostic_table,
                output_dir / f"{lepton}_slice_sigma_{method}_vs_etcone.png",
                x_column="etcone_max",
                value_column=f"sigma_pb_{method}",
                fixed_column="ptcone_max",
                title=f"{lepton}: sigma ({method}) vs etcone at fixed ptcone",
            )

    monotonicity = build_monotonicity_diagnostics(
        diagnostic_table=diagnostic_table,
        tolerance=float(SETTINGS["ISO_SCAN"]["MONOTONICITY_TOL_ABS"]),
    )
    monotonicity["delta_table"].to_csv(output_dir / f"{lepton}_monotonicity_deltas.csv", index=False)
    monotonicity["classification_table"].to_csv(
        output_dir / f"{lepton}_monotonicity_classification.csv",
        index=False,
    )

    summary_lines = [
        "Scan diagnostics are exploratory outputs, not an official isolation systematic.",
        "",
        "Sigma ranges across the scanned grid:",
    ]
    for method in DD_METHODS:
        value_column = f"sigma_pb_{method}"
        summary_lines.append(
            f"{method}: min={diagnostic_table[value_column].min():.6f} pb, max={diagnostic_table[value_column].max():.6f} pb"
        )
    summary_lines.extend(
        [
            "",
            "Monotonicity summary:",
            monotonicity_summary_text(monotonicity["classification_table"]),
        ]
    )
    summary_text = "\n".join(summary_lines)
    write_text(output_dir / f"{lepton}_scan_diagnostics_summary.txt", summary_text)
    return summary_text


def run_channel(lepton: str, backend: dict, output_dir: Path) -> None:
    channel = CHANNELS[lepton]
    output_dir.mkdir(parents=True, exist_ok=True)

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
    iso_dir = output_dir / "iso_scan"
    iso_dir.mkdir(parents=True, exist_ok=True)

    full_scan = None
    allowed_scan = None
    best_point = None
    if SETTINGS["ISO_SCAN"]["RUN_OPTIMISATION_SCAN"] or SETTINGS["ISO_SCAN"]["RUN_SCAN_DIAGNOSTICS"]:
        full_scan, allowed_scan, best_point = scan_isolation(
            plot_os=plot_os,
            plot_ss=plot_ss,
            mass_window=mass_window,
            require_both=require_both,
            ptcone_range=tuple(SETTINGS["ISO_SCAN"]["PTCONE_RANGE"]),
            ptcone_step=float(SETTINGS["ISO_SCAN"]["PTCONE_STEP"]),
            etcone_range=tuple(SETTINGS["ISO_SCAN"]["ETCONE_RANGE"]),
            etcone_step=float(SETTINGS["ISO_SCAN"]["ETCONE_STEP"]),
            os_sig_eff_min=iso_eff_threshold_for(lepton),
            progress_label=lepton,
        )
        full_scan.to_csv(iso_dir / f"{lepton}_iso_scan_full.csv", index=False)
        allowed_scan.to_csv(iso_dir / f"{lepton}_iso_scan_allowed.csv", index=False)
        best_point.to_frame("best").to_csv(iso_dir / f"{lepton}_iso_scan_best.csv")

    if SETTINGS["ISO_SCAN"]["RUN_OPTIMISATION_SCAN"] and best_point is not None:
        best_ptcone = float(best_point["ptcone_max"])
        best_etcone = float(best_point["etcone_max"])
        if SETTINGS["ISO_SCAN"]["SAVE_HEATMAPS"]:
            save_scan_heatmap(
                full_scan,
                iso_dir / f"{lepton}_heatmap_significance_optimisation.png",
                value_column="significance",
                title=f"{lepton}: optimisation significance",
                best_ptcone=best_ptcone,
                best_etcone=best_etcone,
            )
    else:
        best_ptcone = float(SETTINGS["FIXED_ISO"]["ptcone_max"])
        best_etcone = float(SETTINGS["FIXED_ISO"]["etcone_max"])

    plot_os_after = select_plotdict(
        plot_os,
        mass_window=mass_window,
        ptcone_max=best_ptcone,
        etcone_max=best_etcone,
        require_both=require_both,
    )
    plot_ss_after = select_plotdict(
        plot_ss,
        mass_window=mass_window,
        ptcone_max=best_ptcone,
        etcone_max=best_etcone,
        require_both=require_both,
    )

    after_dir = output_dir / "plots_after"
    after_paths = make_after_plots(
        lepton,
        plot_os_after,
        plot_ss_after,
        mass_window,
        best_ptcone,
        best_etcone,
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
    produced_cache: dict[str, float] = {}

    cut_point_result = evaluate_cut_point(
        lepton=lepton,
        plot_os=plot_os,
        plot_ss=plot_ss,
        channel_config=channel,
        produced_event_count_fn=produced_event_count,
        backend=backend,
        mass_window=mass_window,
        ptcone_max=best_ptcone,
        etcone_max=best_etcone,
        require_both=require_both,
        control_cache=control_cache,
        produced_cache=produced_cache,
        include_cutflows=True,
    )
    dd_report = save_cut_point_report(lepton, cut_point_result, output_dir / "additional_data_driven_bkg")

    scan_diagnostic_summary = None
    if SETTINGS["ISO_SCAN"]["RUN_SCAN_DIAGNOSTICS"]:
        if full_scan is None:
            raise RuntimeError("Scan diagnostics requested but no scan grid was built.")

        def diagnostic_evaluator(ptcone_value: float, etcone_value: float) -> dict:
            return evaluate_cut_point(
                lepton=lepton,
                plot_os=plot_os,
                plot_ss=plot_ss,
                channel_config=channel,
                produced_event_count_fn=produced_event_count,
                backend=backend,
                mass_window=mass_window,
                ptcone_max=ptcone_value,
                etcone_max=etcone_value,
                require_both=require_both,
                control_cache=control_cache,
                produced_cache=produced_cache,
                include_cutflows=False,
            )

        diagnostic_table = build_scan_diagnostics_table(full_scan, diagnostic_evaluator)
        scan_diagnostic_summary = _save_scan_diagnostics_outputs(
            lepton=lepton,
            diagnostic_table=diagnostic_table,
            best_ptcone=best_ptcone,
            best_etcone=best_etcone,
            output_dir=iso_dir,
        )

    systematics_result = evaluate_systematics(
        lepton=lepton,
        plot_os=plot_os,
        plot_ss=plot_ss,
        channel_config=channel,
        produced_event_count_fn=produced_event_count,
        backend=backend,
        nominal_cut_point=cut_point_result,
        mass_window=mass_window,
        ptcone_max=best_ptcone,
        etcone_max=best_etcone,
        require_both=require_both,
        systematics_dir=output_dir / "systematics",
        control_cache=control_cache,
        produced_cache=produced_cache,
    )

    uncertainty_table = pd.DataFrame(systematics_result["mass_window"]["uncertainty_summary_by_method"])

    summary_lines = [
        f"Channel: {lepton}",
        "",
        "1. Best isolation working point",
        f"best_ptcone = {best_ptcone:.4f}",
        f"best_etcone = {best_etcone:.4f}",
        "",
        "2. OS cutflow",
        cut_point_result["os_cutflow"].to_string(index=False),
        "",
        "3. SS cutflow",
        cut_point_result["ss_cutflow"].to_string(index=False) if cut_point_result["ss_cutflow"] is not None else "N/A",
        "",
        "4. Nominal cross section summary",
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
    ]
    if scan_diagnostic_summary is not None:
        summary_lines.extend(["", "9. Scan diagnostic / monotonicity summary", scan_diagnostic_summary])

    write_text(output_dir / f"{lepton}_summary.txt", "\n".join(summary_lines))
    write_json(
        output_dir / f"{lepton}_cross_section.json",
        {
            "channel": lepton,
            "best_ptcone": best_ptcone,
            "best_etcone": best_etcone,
            "mass_window": list(mass_window),
            "require_both_iso": require_both,
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


def main() -> None:
    ensure_script_directory()
    ensure_environment()
    backend = import_backend()

    run_root = (Path(__file__).resolve().parent / SETTINGS["OUTPUT_DIR"] / f"run_{now_stamp()}").resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(run_root / "settings.json", SETTINGS)

    for lepton in SETTINGS["LEPTONS"]:
        run_channel(lepton, backend, run_root / lepton)

    print(f"All done. Outputs saved under: {run_root}")


if __name__ == "__main__":
    main()
