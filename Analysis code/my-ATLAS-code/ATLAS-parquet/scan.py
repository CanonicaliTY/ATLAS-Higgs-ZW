from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from config import DD_METHODS
from cross_section import compute_significance
from visualisation import select_plotdict, summarize

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def scan_isolation(
    plot_os: dict,
    plot_ss: dict,
    mass_window: tuple[float, float],
    require_both: bool,
    ptcone_range: tuple[float, float],
    ptcone_step: float,
    etcone_range: tuple[float, float],
    etcone_step: float,
    os_sig_eff_min: float,
    progress_label: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    os_reference = select_plotdict(plot_os, mass_window=mass_window, require_both=require_both)
    ss_reference = select_plotdict(plot_ss, mass_window=mass_window, require_both=require_both)

    _, os_signal_reference, _, _, _ = summarize(os_reference)
    ss_data_reference, _, _, _, _ = summarize(ss_reference)

    ptcone_values = np.arange(ptcone_range[0], ptcone_range[1] + 1e-12, ptcone_step)
    etcone_values = np.arange(etcone_range[0], etcone_range[1] + 1e-12, etcone_step)
    total_points = int(len(ptcone_values) * len(etcone_values))

    iterator = ((float(ptcone), float(etcone)) for ptcone in ptcone_values for etcone in etcone_values)
    if tqdm is not None:
        iterator = tqdm(iterator, total=total_points, desc=f"{progress_label or 'iso'} iso scan", unit="pt")

    rows = []
    for ptcone_max, etcone_max in iterator:
        os_selected = select_plotdict(
            os_reference,
            ptcone_max=ptcone_max,
            etcone_max=etcone_max,
            require_both=require_both,
        )
        ss_selected = select_plotdict(
            ss_reference,
            ptcone_max=ptcone_max,
            etcone_max=etcone_max,
            require_both=require_both,
        )

        os_data, os_signal, os_background, os_signal_over_background, os_data_over_mc = summarize(os_selected)
        ss_data, _, _, _, _ = summarize(ss_selected)

        os_signal_efficiency = (os_signal / os_signal_reference) if os_signal_reference > 0 else float("nan")
        ss_rejection = 1.0 - (ss_data / ss_data_reference) if ss_data_reference > 0 else float("nan")
        significance = compute_significance(os_signal, os_background)

        rows.append(
            {
                "ptcone_max": ptcone_max,
                "etcone_max": etcone_max,
                "significance": significance,
                "OS_sig_eff": os_signal_efficiency,
                "SS_rejection": ss_rejection,
                "OS_Data": os_data,
                "OS_Signal": os_signal,
                "OS_Background": os_background,
                "OS_Data/MC": os_data_over_mc,
                "OS_S/B": os_signal_over_background,
                "SS_Data": ss_data,
            }
        )

    full_scan = pd.DataFrame(rows)
    allowed_scan = full_scan[full_scan["OS_sig_eff"] >= os_sig_eff_min].copy()
    allowed_scan["tightness"] = allowed_scan["ptcone_max"] + allowed_scan["etcone_max"]
    allowed_scan = allowed_scan.sort_values(
        ["significance", "SS_rejection", "tightness"],
        ascending=[False, False, True],
    )
    if allowed_scan.empty:
        raise RuntimeError(
            f"No scan point satisfies OS_sig_eff_min={os_sig_eff_min}. "
            "Try loosening ISO_SCAN ranges or lowering OS_SIG_EFF_MIN."
        )

    best_point = allowed_scan.iloc[0]
    return full_scan, allowed_scan, best_point


def build_scan_diagnostics_table(
    scan_table: pd.DataFrame,
    cut_point_evaluator: Callable[[float, float], dict],
) -> pd.DataFrame:
    rows = []
    for row in scan_table.to_dict(orient="records"):
        evaluation = cut_point_evaluator(float(row["ptcone_max"]), float(row["etcone_max"]))
        sigma_results = evaluation["sigma_results_table"].set_index("method")
        rows.append(
            {
                **row,
                "extra_bkg_wrong_flavour": float(sigma_results.loc["wrong_flavour", "extra_bkg"]),
                "extra_bkg_wrong_charge": float(sigma_results.loc["wrong_charge", "extra_bkg"]),
                "extra_bkg_both_average": float(sigma_results.loc["both_average", "extra_bkg"]),
                "sigma_pb_none": float(sigma_results.loc["none", "sigma_pb"]),
                "sigma_pb_wrong_flavour": float(sigma_results.loc["wrong_flavour", "sigma_pb"]),
                "sigma_pb_wrong_charge": float(sigma_results.loc["wrong_charge", "sigma_pb"]),
                "sigma_pb_both_average": float(sigma_results.loc["both_average", "sigma_pb"]),
                "sigma_shift_pb_wrong_flavour": float(sigma_results.loc["wrong_flavour", "sigma_shift_pb"]),
                "sigma_shift_pb_wrong_charge": float(sigma_results.loc["wrong_charge", "sigma_shift_pb"]),
                "sigma_shift_pb_both_average": float(sigma_results.loc["both_average", "sigma_shift_pb"]),
            }
        )
    return pd.DataFrame(rows)


def classify_monotonic_deltas(deltas: list[float], tolerance: float) -> str:
    if len(deltas) == 0:
        return "inconclusive / too few points"

    signs = []
    for delta in deltas:
        if delta > tolerance:
            signs.append(1)
        elif delta < -tolerance:
            signs.append(-1)
        else:
            signs.append(0)

    non_zero = {sign for sign in signs if sign != 0}
    if not non_zero:
        return "inconclusive / too few points"
    if non_zero == {1}:
        return "monotonic increasing"
    if non_zero == {-1}:
        return "monotonic decreasing"
    return "non-monotonic"


def build_monotonicity_diagnostics(
    diagnostic_table: pd.DataFrame,
    tolerance: float,
) -> dict[str, pd.DataFrame]:
    delta_rows = []
    classification_rows = []
    sigma_columns = [f"sigma_pb_{method}" for method in DD_METHODS]

    for sigma_column in sigma_columns:
        method = sigma_column.removeprefix("sigma_pb_")
        for fixed_axis, varying_axis in (("etcone_max", "ptcone_max"), ("ptcone_max", "etcone_max")):
            for fixed_value, slice_frame in diagnostic_table.groupby(fixed_axis):
                ordered = slice_frame.sort_values(varying_axis)
                values = ordered[sigma_column].tolist()
                varying_values = ordered[varying_axis].tolist()
                deltas = []
                for index in range(len(values) - 1):
                    delta = float(values[index + 1] - values[index])
                    deltas.append(delta)
                    delta_rows.append(
                        {
                            "method": method,
                            "fixed_axis": fixed_axis,
                            "fixed_value": float(fixed_value),
                            "varying_axis": varying_axis,
                            "varying_start": float(varying_values[index]),
                            "varying_end": float(varying_values[index + 1]),
                            "delta": delta,
                        }
                    )

                classification_rows.append(
                    {
                        "method": method,
                        "fixed_axis": fixed_axis,
                        "fixed_value": float(fixed_value),
                        "varying_axis": varying_axis,
                        "classification": classify_monotonic_deltas(deltas, tolerance),
                        "n_points": len(values),
                    }
                )

    delta_table = pd.DataFrame(delta_rows)
    classification_table = pd.DataFrame(classification_rows)
    return {"delta_table": delta_table, "classification_table": classification_table}


def monotonicity_summary_text(classification_table: pd.DataFrame) -> str:
    if classification_table.empty:
        return "No monotonicity slices were available."

    lines = []
    for method in DD_METHODS:
        method_rows = classification_table[classification_table["method"] == method]
        if method_rows.empty:
            continue
        lines.append(f"Method: {method}")
        counts = method_rows.groupby(["fixed_axis", "classification"]).size().reset_index(name="count")
        lines.append(counts.to_string(index=False))
        lines.append("")
    return "\n".join(lines).strip()

