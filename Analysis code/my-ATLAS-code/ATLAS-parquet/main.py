from __future__ import annotations

from pathlib import Path

from config import CHANNELS, LUMI_REL_UNC, SETTINGS
from control_regions import build_additional_background_report
from parquet_io import load_main_events
from scan import scan_isolation
from systematics import evaluate_systematics
from utils import ensure_environment, ensure_script_directory, import_backend, now_stamp, write_json, write_text
from visualisation import (
    build_plot_dict,
    compose_image_grid,
    cutflow_table,
    make_after_plots,
    make_before_plots,
    save_scan_heatmap,
    select_plotdict,
)


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

    full_scan = None
    allowed_scan = None
    best_point = None
    iso_dir = output_dir / "iso_scan"
    iso_dir.mkdir(parents=True, exist_ok=True)

    if SETTINGS["USE_SCAN"]:
        full_scan, allowed_scan, best_point = scan_isolation(
            plot_os=plot_os,
            plot_ss=plot_ss,
            mass_window=mass_window,
            require_both=require_both,
            ptcone_range=tuple(SETTINGS["ISO_SCAN"]["PTCONE_RANGE"]),
            ptcone_step=float(SETTINGS["ISO_SCAN"]["PTCONE_STEP"]),
            etcone_range=tuple(SETTINGS["ISO_SCAN"]["ETCONE_RANGE"]),
            etcone_step=float(SETTINGS["ISO_SCAN"]["ETCONE_STEP"]),
            os_sig_eff_min=float(SETTINGS["ISO_SCAN"]["OS_SIG_EFF_MIN"][lepton])
            if isinstance(SETTINGS["ISO_SCAN"]["OS_SIG_EFF_MIN"], dict)
            else float(SETTINGS["ISO_SCAN"]["OS_SIG_EFF_MIN"]),
            progress_label=lepton,
        )
        best_ptcone = float(best_point["ptcone_max"])
        best_etcone = float(best_point["etcone_max"])

        if SETTINGS["SAVE_SCAN_TABLES"]:
            full_scan.to_csv(iso_dir / f"{lepton}_iso_scan_full.csv", index=False)
            allowed_scan.to_csv(iso_dir / f"{lepton}_iso_scan_allowed.csv", index=False)
            best_point.to_frame("best").to_csv(iso_dir / f"{lepton}_iso_scan_best.csv")

        if SETTINGS["SAVE_PLOTS"]:
            save_scan_heatmap(
                full_scan,
                iso_dir / f"{lepton}_heatmap_SS_rejection.png",
                value_column="SS_rejection",
                title=f"{lepton}: SS rejection",
                best_ptcone=best_ptcone,
                best_etcone=best_etcone,
            )
            save_scan_heatmap(
                full_scan,
                iso_dir / f"{lepton}_heatmap_OS_sig_eff.png",
                value_column="OS_sig_eff",
                title=f"{lepton}: OS signal efficiency",
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

    os_cutflow = cutflow_table(
        plot_os,
        mass_window=mass_window,
        ptcone_max=best_ptcone,
        etcone_max=best_etcone,
        require_both=require_both,
    )
    ss_cutflow = cutflow_table(
        plot_ss,
        mass_window=mass_window,
        ptcone_max=best_ptcone,
        etcone_max=best_etcone,
        require_both=require_both,
    )
    if SETTINGS["SAVE_TABLES"]:
        os_cutflow.to_csv(output_dir / f"{lepton}_cutflow_OS.csv", index=False)
        ss_cutflow.to_csv(output_dir / f"{lepton}_cutflow_SS.csv", index=False)

    control_cache: dict = {}
    produced_cache: dict[str, float] = {}

    additional_report = build_additional_background_report(
        lepton=lepton,
        plot_os=plot_os,
        channel_config=channel,
        produced_event_count_fn=produced_event_count,
        backend=backend,
        mass_window=mass_window,
        ptcone_max=best_ptcone,
        etcone_max=best_etcone,
        require_both=require_both,
        output_dir=output_dir / "additional_data_driven_bkg",
        control_cache=control_cache,
        produced_cache=produced_cache,
    )

    nominal_sigma = additional_report["sigma_with_additional_bkg"]
    sigma_without_additional_bkg = additional_report["sigma_without_additional_bkg"]

    systematics_result = evaluate_systematics(
        lepton=lepton,
        plot_os=plot_os,
        channel_config=channel,
        produced_event_count_fn=produced_event_count,
        backend=backend,
        mass_window=mass_window,
        ptcone_max=best_ptcone,
        etcone_max=best_etcone,
        require_both=require_both,
        full_scan=full_scan,
        allowed_scan=allowed_scan,
        systematics_dir=output_dir / "systematics",
        scan_output_dir=iso_dir,
        control_cache=control_cache,
        produced_cache=produced_cache,
    )
    total_systematic_pb = float(systematics_result["total_systematic_pb"])
    lumi_pb = LUMI_REL_UNC * nominal_sigma["sigma_pb"]

    summary_lines = [
        f"Channel: {lepton}",
        f"Nominal method: {additional_report['selected_method']}",
        f"Nominal order mode: {additional_report['selected_order_mode']}",
        f"Mass window [GeV]: {mass_window[0]:.1f} to {mass_window[1]:.1f}",
        f"Isolation working point: ptcone<{best_ptcone:.3f}, etcone<{best_etcone:.3f}, require_both={require_both}",
        "",
        "OS cutflow:",
        os_cutflow.to_string(index=False),
        "",
        "SS cutflow:",
        ss_cutflow.to_string(index=False),
        "",
        f"Sigma without additional background [pb]: {sigma_without_additional_bkg['sigma_pb']:.6f}",
        f"Sigma with selected estimator [pb]: {nominal_sigma['sigma_pb']:.6f}",
        f"Applied additional background events: {additional_report['applied_extra_background']:.6f}",
        (
            f"Quoted result [pb]: {nominal_sigma['sigma_pb']:.6f} "
            f"+/- {nominal_sigma['dsigma_stat_pb']:.6f} (stat.) "
            f"+/- {total_systematic_pb:.6f} (syst.) "
            f"+/- {lumi_pb:.6f} (lumi.)"
        ),
    ]
    write_text(output_dir / f"{lepton}_summary.txt", "\n".join(summary_lines))

    write_json(
        output_dir / f"{lepton}_cross_section.json",
        {
            "channel": lepton,
            "best_ptcone": best_ptcone,
            "best_etcone": best_etcone,
            "mass_window": list(mass_window),
            "require_both_iso": require_both,
            "sigma_without_additional_bkg": sigma_without_additional_bkg,
            "nominal": nominal_sigma,
            "additional_data_driven_bkg": additional_report,
            "systematics": systematics_result,
            "lumi_pb": lumi_pb,
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
