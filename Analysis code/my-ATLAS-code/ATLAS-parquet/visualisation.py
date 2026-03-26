from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path(tempfile.gettempdir()) / "atlas_parquet_mplconfig").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from config import CHANNELS, SETTINGS, user_facing_sample_label
from selections import apply_selection
from utils import ensure_parent, get_sample_key_by_prefix, yield_data, yield_mc


def build_plot_dict(events_by_sample: dict, lepton: str) -> dict:
    channel = CHANNELS[lepton]
    plot_dict: dict[str, object] = {}

    data_key = get_sample_key_by_prefix(events_by_sample, "2to4lep")
    plot_dict["Data"] = events_by_sample.get(data_key) if data_key else None

    for sample_code in channel["signal_samples"]:
        sample_key = get_sample_key_by_prefix(events_by_sample, sample_code)
        label = f"Signal {user_facing_sample_label(sample_code)}"
        plot_dict[label] = events_by_sample.get(sample_key) if sample_key else None

    for sample_code in channel["background_samples"]:
        sample_key = get_sample_key_by_prefix(events_by_sample, sample_code)
        plot_dict[f"Background {sample_code}"] = events_by_sample.get(sample_key) if sample_key else None

    return plot_dict


def select_plotdict(plot_dict: dict, **selection_kwargs) -> dict:
    return {label: apply_selection(events, **selection_kwargs) for label, events in plot_dict.items()}


def summarize(plot_dict: dict) -> tuple[float, float, float, float, float]:
    n_data = yield_data(plot_dict.get("Data"))
    n_signal = 0.0
    n_background = 0.0
    for label, events in plot_dict.items():
        if label.startswith("Signal"):
            n_signal += yield_mc(events)
        elif label.startswith("Background"):
            n_background += yield_mc(events)

    n_mc = n_signal + n_background
    signal_over_background = (n_signal / n_background) if n_background > 0 else float("inf")
    data_over_mc = (n_data / n_mc) if n_mc > 0 else float("nan")
    return n_data, n_signal, n_background, signal_over_background, data_over_mc


def cutflow_table(
    plot_dict: dict,
    mass_window: tuple[float, float],
    ptcone_max: float,
    etcone_max: float,
    require_both: bool = True,
) -> pd.DataFrame:
    steps = [
        ("Baseline", dict()),
        ("Mass window", dict(mass_window=mass_window)),
        ("Iso only", dict(ptcone_max=ptcone_max, etcone_max=etcone_max)),
        ("Mass + Iso", dict(mass_window=mass_window, ptcone_max=ptcone_max, etcone_max=etcone_max)),
    ]
    rows = []
    for step_name, selection_kwargs in steps:
        selected = select_plotdict(plot_dict, require_both=require_both, **selection_kwargs)
        n_data, n_signal, n_background, s_over_b, data_over_mc = summarize(selected)
        rows.append([step_name, n_data, n_signal, n_background, s_over_b, data_over_mc])
    return pd.DataFrame(rows, columns=["Step", "Data", "Signal", "Background", "S/B", "Data/MC"])


def save_fig(figure: plt.Figure, output_path: Path, dpi: int = 150) -> None:
    ensure_parent(output_path)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def compose_image_grid(
    image_paths: list[Path],
    output_path: Path,
    nrows: int,
    ncols: int,
    titles: list[str] | None = None,
    figsize: tuple[float, float] = (14, 10),
) -> None:
    ensure_parent(output_path)
    figure, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    for index, axis in enumerate(axes):
        if index >= len(image_paths) or not image_paths[index].exists():
            axis.axis("off")
            continue
        axis.imshow(plt.imread(image_paths[index]))
        axis.axis("off")
        if titles and index < len(titles):
            axis.set_title(titles[index], fontsize=12)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def make_before_plots(lepton: str, plot_os: dict, plot_ss: dict, plot_stacked_hist, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    channel = CHANNELS[lepton]
    colour_list = ["k", "b", "y", "g", "r", "m"]

    plot_paths: list[Path] = []

    pt_config = SETTINGS["PLOTS"]["LEADING_PT"]
    fig_os_pt, _ = plot_stacked_hist(
        plot_os,
        "lep_pt[0]",
        colour_list,
        pt_config["bins"],
        pt_config["xmin"],
        pt_config["xmax"],
        f"{channel['leading_label']} ({lepton} OS, before cuts)",
        logy=pt_config["logy"],
        show_text=True,
        residual_plot=True,
        save_fig=False,
    )
    path_os_pt = output_dir / f"{lepton}_OS_leadingPt_before.png"
    if SETTINGS["SAVE_PLOTS"]:
        save_fig(fig_os_pt, path_os_pt)
    plot_paths.append(path_os_pt)

    fig_ss_pt, _ = plot_stacked_hist(
        plot_ss,
        "lep_pt[0]",
        colour_list,
        pt_config["bins"],
        pt_config["xmin"],
        pt_config["xmax"],
        f"{channel['leading_label']} ({lepton} SS, before cuts)",
        logy=pt_config["logy"],
        show_text=True,
        residual_plot=True,
        save_fig=False,
    )
    path_ss_pt = output_dir / f"{lepton}_SS_leadingPt_before.png"
    if SETTINGS["SAVE_PLOTS"]:
        save_fig(fig_ss_pt, path_ss_pt)
    plot_paths.append(path_ss_pt)

    mass_config = SETTINGS["PLOTS"]["MASS_FULL"]
    fig_os_mass, _ = plot_stacked_hist(
        plot_os,
        "mass",
        colour_list,
        mass_config["bins"],
        mass_config["xmin"],
        mass_config["xmax"],
        f"mass [GeV] ({lepton} OS, before cuts)",
        logy=mass_config["logy"],
        show_text=True,
        residual_plot=True,
        save_fig=False,
    )
    path_os_mass = output_dir / f"{lepton}_OS_mass_before.png"
    if SETTINGS["SAVE_PLOTS"]:
        save_fig(fig_os_mass, path_os_mass)
    plot_paths.append(path_os_mass)

    fig_ss_mass, _ = plot_stacked_hist(
        plot_ss,
        "mass",
        colour_list,
        mass_config["bins"],
        mass_config["xmin"],
        mass_config["xmax"],
        f"mass [GeV] ({lepton} SS, before cuts)",
        logy=mass_config["logy"],
        show_text=True,
        residual_plot=True,
        save_fig=False,
    )
    path_ss_mass = output_dir / f"{lepton}_SS_mass_before.png"
    if SETTINGS["SAVE_PLOTS"]:
        save_fig(fig_ss_mass, path_ss_mass)
    plot_paths.append(path_ss_mass)

    return plot_paths


def make_after_plots(
    lepton: str,
    plot_os_after: dict,
    plot_ss_after: dict,
    mass_window: tuple[float, float],
    ptcone_max: float,
    etcone_max: float,
    plot_stacked_hist,
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    colour_list = ["k", "b", "y", "g", "r", "m"]
    zoom_config = SETTINGS["PLOTS"]["MASS_ZOOM"]

    figure_os, _ = plot_stacked_hist(
        plot_os_after,
        "mass",
        colour_list,
        zoom_config["bins"],
        zoom_config["xmin"],
        zoom_config["xmax"],
        (
            f"mass [GeV] ({lepton} OS after cuts: "
            f"{mass_window[0]:.1f}<m<{mass_window[1]:.1f}, "
            f"ptcone<{ptcone_max:.3f}, etcone<{etcone_max:.3f})"
        ),
        logy=zoom_config["logy"],
        show_text=True,
        residual_plot=True,
        save_fig=False,
    )
    path_os = output_dir / f"{lepton}_OS_mass_after.png"
    if SETTINGS["SAVE_PLOTS"]:
        save_fig(figure_os, path_os)

    figure_ss, _ = plot_stacked_hist(
        plot_ss_after,
        "mass",
        colour_list,
        zoom_config["bins"],
        zoom_config["xmin"],
        zoom_config["xmax"],
        (
            f"mass [GeV] ({lepton} SS after cuts: "
            f"{mass_window[0]:.1f}<m<{mass_window[1]:.1f}, "
            f"ptcone<{ptcone_max:.3f}, etcone<{etcone_max:.3f})"
        ),
        logy=zoom_config["logy"],
        show_text=True,
        residual_plot=True,
        save_fig=False,
    )
    path_ss = output_dir / f"{lepton}_SS_mass_after.png"
    if SETTINGS["SAVE_PLOTS"]:
        save_fig(figure_ss, path_ss)

    return [path_os, path_ss]


def save_scan_heatmap(
    scan_table: pd.DataFrame,
    output_path: Path,
    value_column: str,
    title: str,
    value_label: str | None = None,
    nominal_ptcone: float | None = None,
    nominal_etcone: float | None = None,
) -> None:
    pivot = scan_table.pivot(index="etcone_max", columns="ptcone_max", values=value_column)
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)
    values = np.asarray(pivot.values, dtype=float)
    masked = np.ma.masked_invalid(values)

    figure, axis = plt.subplots(figsize=(8, 6))
    if masked.count() == 0:
        axis.text(0.5, 0.5, "No valid values", ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
        save_fig(figure, output_path)
        return

    image = axis.imshow(masked, origin="lower", aspect="auto")
    axis.set_title(title)
    axis.set_xlabel("ptcone_max [GeV]")
    axis.set_ylabel("etcone_max [GeV]")

    x_ticks = np.linspace(0, pivot.shape[1] - 1, min(6, pivot.shape[1])).astype(int)
    y_ticks = np.linspace(0, pivot.shape[0] - 1, min(6, pivot.shape[0])).astype(int)
    axis.set_xticks(x_ticks)
    axis.set_yticks(y_ticks)
    axis.set_xticklabels([f"{pivot.columns[index]:.2f}" for index in x_ticks], rotation=45, ha="right")
    axis.set_yticklabels([f"{pivot.index[index]:.2f}" for index in y_ticks])
    figure.colorbar(image, ax=axis, label=value_label or value_column)

    if nominal_ptcone is not None and nominal_etcone is not None:
        try:
            x_position = list(pivot.columns).index(nominal_ptcone)
            y_position = list(pivot.index).index(nominal_etcone)
            axis.scatter([x_position], [y_position], marker="x", s=100, color="white")
        except ValueError:
            pass

    save_fig(figure, output_path)


def save_additional_bkg_plot(table: pd.DataFrame, output_path: Path, title: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    x_values = np.arange(len(table))
    width = 0.25
    axis.bar(x_values - width, table["N_data"].values, width=width, label="N_data")
    axis.bar(x_values, table["N_MC"].values, width=width, label="N_MC")
    axis.bar(x_values + width, table["residual"].values, width=width, label="residual")
    axis.axhline(0.0, color="black", linewidth=1.0)
    axis.set_xticks(x_values)
    axis.set_xticklabels(table["region"].tolist(), rotation=15, ha="right")
    axis.set_ylabel("events")
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    save_fig(figure, output_path)


def save_estimator_plot(table: pd.DataFrame, output_path: Path, title: str) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    x_values = np.arange(len(table))
    width = 0.35
    axis.bar(x_values - width / 2.0, table["estimate_raw"].values, width=width, label="estimate_raw")
    axis.bar(x_values + width / 2.0, table["estimate_clipped"].values, width=width, label="estimate_clipped")
    axis.axhline(0.0, color="black", linewidth=1.0)
    axis.set_xticks(x_values)
    axis.set_xticklabels(table["method"].tolist(), rotation=15, ha="right")
    axis.set_ylabel("estimated additional background events")
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    save_fig(figure, output_path)


def save_surface_plot(
    scan_table: pd.DataFrame,
    output_path: Path,
    value_column: str,
    title: str,
    value_label: str | None = None,
    nominal_ptcone: float | None = None,
    nominal_etcone: float | None = None,
) -> None:
    pivot = scan_table.pivot(index="etcone_max", columns="ptcone_max", values=value_column)
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)

    x_grid, y_grid = np.meshgrid(pivot.columns.values, pivot.index.values)
    z_grid = np.ma.masked_invalid(np.asarray(pivot.values, dtype=float))

    figure = plt.figure(figsize=(9, 7))
    axis = figure.add_subplot(111, projection="3d")
    if z_grid.count() == 0:
        axis.text2D(0.5, 0.5, "No valid values", ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
        save_fig(figure, output_path)
        return
    surface = axis.plot_surface(x_grid, y_grid, z_grid, cmap="viridis", edgecolor="none")
    axis.set_xlabel("ptcone_max [GeV]")
    axis.set_ylabel("etcone_max [GeV]")
    axis.set_zlabel(value_label or value_column)
    axis.set_title(title)
    if nominal_ptcone is not None and nominal_etcone is not None:
        nominal_rows = scan_table[
            np.isclose(scan_table["ptcone_max"].to_numpy(dtype=float), float(nominal_ptcone))
            & np.isclose(scan_table["etcone_max"].to_numpy(dtype=float), float(nominal_etcone))
        ]
        if not nominal_rows.empty:
            nominal_value = float(nominal_rows.iloc[0][value_column])
            if np.isfinite(nominal_value):
                axis.scatter(
                    [float(nominal_ptcone)],
                    [float(nominal_etcone)],
                    [nominal_value],
                    color="black",
                    s=35,
                )
    figure.colorbar(surface, ax=axis, shrink=0.65, pad=0.1)
    save_fig(figure, output_path)


def save_slice_plot(
    scan_table: pd.DataFrame,
    output_path: Path,
    x_column: str,
    value_column: str,
    fixed_column: str,
    title: str,
    value_label: str | None = None,
    fixed_value: float | None = None,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    grouped = list(scan_table.groupby(fixed_column))
    if fixed_value is not None and grouped:
        available = np.asarray([float(value) for value, _ in grouped], dtype=float)
        chosen_index = int(np.argmin(np.abs(available - float(fixed_value))))
        grouped = [grouped[chosen_index]]

    for fixed_axis_value, slice_frame in grouped:
        ordered = slice_frame.sort_values(x_column)
        axis.plot(
            ordered[x_column].values,
            ordered[value_column].values,
            marker="o",
            label=f"{fixed_column}={float(fixed_axis_value):.2f}",
        )
    axis.set_xlabel(f"{x_column} [GeV]")
    axis.set_ylabel(value_label or value_column)
    axis.set_title(title)
    axis.legend(fontsize=8, ncols=2)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    save_fig(figure, output_path)
