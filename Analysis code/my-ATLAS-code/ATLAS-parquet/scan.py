from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from config import DD_METHODS
from utils import progress_iter
from visualisation import select_plotdict, summarize


def _round_iso_value(value: float) -> float:
    return round(float(value), 6)


def _inclusive_axis_values(value_range: tuple[float, float], step: float) -> list[float]:
    if step <= 0:
        raise ValueError(f"Scan step must be positive, got {step}")
    start, stop = float(value_range[0]), float(value_range[1])
    if stop < start:
        raise ValueError(f"Scan range must satisfy stop >= start, got {value_range}")
    values = np.arange(start, stop + 0.5 * step, step, dtype=float)
    return [_round_iso_value(value) for value in values.tolist()]


def _local_axis_values(
    nominal_value: float,
    half_width: float,
    step: float,
    value_range: tuple[float, float],
) -> list[float]:
    if half_width < 0:
        raise ValueError(f"Local-box half width must be non-negative, got {half_width}")
    n_steps = int(np.floor((half_width / step) + 1e-12))
    offsets = np.arange(-n_steps, n_steps + 1, dtype=float) * float(step)
    values = [_round_iso_value(float(nominal_value) + offset) for offset in offsets.tolist()]
    values.append(_round_iso_value(nominal_value))
    low, high = float(value_range[0]), float(value_range[1])
    kept = [value for value in values if low <= value <= high]
    if not kept:
        kept = [_round_iso_value(nominal_value)]
    return sorted(dict.fromkeys(kept))


def _diagonal_points(ptcone_values: list[float], etcone_values: list[float]) -> list[tuple[float, float]]:
    if not ptcone_values or not etcone_values:
        return []
    n_points = max(len(ptcone_values), len(etcone_values))
    pt_indices = np.linspace(0, len(ptcone_values) - 1, num=n_points)
    et_indices = np.linspace(0, len(etcone_values) - 1, num=n_points)
    points = []
    for pt_index, et_index in zip(pt_indices, et_indices):
        points.append(
            (
                _round_iso_value(ptcone_values[int(round(pt_index))]),
                _round_iso_value(etcone_values[int(round(et_index))]),
            )
        )
    return list(dict.fromkeys(points))


def generate_scan_points(
    nominal_ptcone: float,
    nominal_etcone: float,
    scan_mode: str,
    ptcone_range: tuple[float, float],
    ptcone_step: float,
    etcone_range: tuple[float, float],
    etcone_step: float,
    local_box_ptcone_half_width: float,
    local_box_etcone_half_width: float,
) -> list[tuple[float, float]]:
    ptcone_values = _inclusive_axis_values(ptcone_range, ptcone_step)
    etcone_values = _inclusive_axis_values(etcone_range, etcone_step)
    nominal_ptcone = _round_iso_value(nominal_ptcone)
    nominal_etcone = _round_iso_value(nominal_etcone)

    mode = str(scan_mode).strip().lower()
    if mode == "full_grid":
        points = [(ptcone, etcone) for ptcone in ptcone_values for etcone in etcone_values]
    elif mode == "diagonal":
        points = _diagonal_points(ptcone_values, etcone_values)
    elif mode == "crosshair_nominal":
        varying_ptcone = sorted(dict.fromkeys(ptcone_values + [nominal_ptcone]))
        varying_etcone = sorted(dict.fromkeys(etcone_values + [nominal_etcone]))
        points = [(ptcone, nominal_etcone) for ptcone in varying_ptcone]
        points.extend((nominal_ptcone, etcone) for etcone in varying_etcone)
    elif mode == "local_box":
        local_ptcone = _local_axis_values(
            nominal_value=nominal_ptcone,
            half_width=local_box_ptcone_half_width,
            step=ptcone_step,
            value_range=ptcone_range,
        )
        local_etcone = _local_axis_values(
            nominal_value=nominal_etcone,
            half_width=local_box_etcone_half_width,
            step=etcone_step,
            value_range=etcone_range,
        )
        points = [(ptcone, etcone) for ptcone in local_ptcone for etcone in local_etcone]
    else:
        raise ValueError(
            f"Unsupported ISO_SCAN['SCAN_MODE']={scan_mode!r}. "
            "Choose from 'full_grid', 'diagonal', 'crosshair_nominal', or 'local_box'."
        )

    return list(dict.fromkeys((_round_iso_value(ptcone), _round_iso_value(etcone)) for ptcone, etcone in points))


def scan_isolation(
    plot_os: dict,
    plot_ss: dict,
    mass_window: tuple[float, float],
    require_both: bool,
    nominal_ptcone: float,
    nominal_etcone: float,
    scan_mode: str,
    ptcone_range: tuple[float, float],
    ptcone_step: float,
    etcone_range: tuple[float, float],
    etcone_step: float,
    local_box_ptcone_half_width: float,
    local_box_etcone_half_width: float,
    progress_label: str | None = None,
) -> pd.DataFrame:
    # The scan is diagnostic only. FIXED_ISO remains the nominal cut used later.
    os_reference = select_plotdict(plot_os, mass_window=mass_window, require_both=require_both)
    ss_reference = select_plotdict(plot_ss, mass_window=mass_window, require_both=require_both)

    _, os_signal_reference, _, _, _ = summarize(os_reference)
    ss_data_reference, _, _, _, _ = summarize(ss_reference)

    scan_points = generate_scan_points(
        nominal_ptcone=nominal_ptcone,
        nominal_etcone=nominal_etcone,
        scan_mode=scan_mode,
        ptcone_range=ptcone_range,
        ptcone_step=ptcone_step,
        etcone_range=etcone_range,
        etcone_step=etcone_step,
        local_box_ptcone_half_width=local_box_ptcone_half_width,
        local_box_etcone_half_width=local_box_etcone_half_width,
    )

    iterator = progress_iter(
        scan_points,
        total=len(scan_points),
        desc=f"{progress_label or 'iso'} scan",
        unit="pt",
    )

    rows = []
    for index, (ptcone_max, etcone_max) in enumerate(iterator):
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

        rows.append(
            {
                "scan_mode": str(scan_mode),
                "scan_order": index,
                "ptcone_max": float(ptcone_max),
                "etcone_max": float(etcone_max),
                "is_nominal_scan_point": bool(
                    np.isclose(ptcone_max, nominal_ptcone) and np.isclose(etcone_max, nominal_etcone)
                ),
                "delta_ptcone_from_nominal": float(ptcone_max - nominal_ptcone),
                "delta_etcone_from_nominal": float(etcone_max - nominal_etcone),
                "distance_to_nominal": float(np.hypot(ptcone_max - nominal_ptcone, etcone_max - nominal_etcone)),
                "OS_sig_eff": os_signal_efficiency,
                "OS_Data": os_data,
                "OS_Signal": os_signal,
                "OS_Background": os_background,
                "OS_Data/MC": os_data_over_mc,
                "OS_S/B": os_signal_over_background,
                "SS_Data": ss_data,
                "SS_rejection": ss_rejection,
            }
        )

    return pd.DataFrame(rows)


def build_scan_diagnostics_table(
    scan_table: pd.DataFrame,
    cut_point_evaluator: Callable[[float, float], dict],
    nominal_sigma_lookup: pd.DataFrame,
    methods: Sequence[str] | None = None,
) -> pd.DataFrame:
    resolved_methods = tuple(method for method in (methods or DD_METHODS) if method in nominal_sigma_lookup.index)
    rows = []
    scan_rows = scan_table.to_dict(orient="records")
    iterator = progress_iter(scan_rows, total=len(scan_rows), desc="scan diagnostics", unit="pt")
    for row in iterator:
        evaluation = cut_point_evaluator(float(row["ptcone_max"]), float(row["etcone_max"]))
        sigma_results = evaluation["sigma_results_table"].set_index("method")
        sigma_valid_series = (
            sigma_results["sigma_valid"]
            if "sigma_valid" in sigma_results.columns
            else pd.Series(True, index=sigma_results.index)
        )
        sigma_error_series = (
            sigma_results["sigma_error"]
            if "sigma_error" in sigma_results.columns
            else pd.Series("", index=sigma_results.index)
        )
        point_valid = bool(sigma_valid_series.fillna(False).all())
        point_errors = [
            str(error)
            for error in sigma_error_series.tolist()
            if isinstance(error, str) and error.strip()
        ]

        row_result = {
            **row,
            "sigma_defined": point_valid,
            "sigma_error": "; ".join(dict.fromkeys(point_errors)),
        }
        for method in resolved_methods:
            if method not in sigma_results.index:
                continue
            sigma_value = float(sigma_results.loc[method, "sigma_pb"])
            nominal_sigma = float(nominal_sigma_lookup.loc[method, "sigma_pb"])
            signed_shift = sigma_value - nominal_sigma if np.isfinite(sigma_value) and np.isfinite(nominal_sigma) else float("nan")
            abs_shift = abs(signed_shift) if np.isfinite(signed_shift) else float("nan")
            frac_shift = (
                abs_shift / abs(nominal_sigma)
                if np.isfinite(abs_shift) and np.isfinite(nominal_sigma) and nominal_sigma != 0.0
                else float("nan")
            )
            row_result[f"extra_bkg_{method}"] = float(sigma_results.loc[method, "extra_bkg"])
            row_result[f"sigma_pb_{method}"] = sigma_value
            row_result[f"sigma_shift_pb_{method}"] = signed_shift
            row_result[f"sigma_abs_shift_pb_{method}"] = abs_shift
            row_result[f"sigma_frac_shift_{method}"] = frac_shift
            if "epsilon" in sigma_results.columns:
                row_result[f"epsilon_{method}"] = float(sigma_results.loc[method, "epsilon"])
        rows.append(row_result)
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
    methods: Sequence[str] | None = None,
) -> dict[str, pd.DataFrame]:
    resolved_methods = tuple(method for method in (methods or DD_METHODS) if f"sigma_pb_{method}" in diagnostic_table.columns)
    delta_rows = []
    classification_rows = []

    for method in resolved_methods:
        sigma_column = f"sigma_pb_{method}"
        for fixed_axis, varying_axis in (("etcone_max", "ptcone_max"), ("ptcone_max", "etcone_max")):
            for fixed_value, slice_frame in diagnostic_table.groupby(fixed_axis):
                ordered = slice_frame.sort_values(varying_axis)
                finite_mask = np.isfinite(ordered[sigma_column].to_numpy(dtype=float))
                ordered = ordered.loc[finite_mask]
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


def build_plateau_diagnostics(
    delta_table: pd.DataFrame,
    tolerance: float,
    methods: Sequence[str] | None = None,
) -> pd.DataFrame:
    resolved_methods = tuple(method for method in (methods or DD_METHODS))
    rows = []
    for method in resolved_methods:
        method_rows = delta_table[delta_table["method"] == method]
        small_delta_count = int(method_rows["delta"].abs().le(tolerance).sum()) if not method_rows.empty else 0
        rows.append(
            {
                "method": method,
                "n_small_deltas": small_delta_count,
                "plateau_like_behaviour": bool(small_delta_count > 0),
                "plateau_assessment": (
                    "some locally flat deltas were seen within the configured tolerance"
                    if small_delta_count > 0
                    else "no clear plateau was seen in the scanned region"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_local_stability_diagnostics(
    diagnostic_table: pd.DataFrame,
    nominal_ptcone: float,
    nominal_etcone: float,
    methods: Sequence[str] | None = None,
    max_neighbours: int = 4,
) -> pd.DataFrame:
    resolved_methods = tuple(
        method for method in (methods or DD_METHODS) if f"sigma_shift_pb_{method}" in diagnostic_table.columns
    )
    if diagnostic_table.empty or max_neighbours <= 0 or not resolved_methods:
        return pd.DataFrame()

    working = diagnostic_table.copy()
    if "distance_to_nominal" not in working.columns:
        working["distance_to_nominal"] = np.hypot(
            working["ptcone_max"].to_numpy(dtype=float) - float(nominal_ptcone),
            working["etcone_max"].to_numpy(dtype=float) - float(nominal_etcone),
        )

    neighbours = working[working["distance_to_nominal"] > 0].copy()
    neighbours = neighbours.sort_values(["distance_to_nominal", "ptcone_max", "etcone_max"]).head(max_neighbours)
    if neighbours.empty:
        return pd.DataFrame()

    rows = []
    for method in resolved_methods:
        for rank, (_, neighbour) in enumerate(neighbours.iterrows(), start=1):
            distance = float(neighbour["distance_to_nominal"])
            abs_shift = float(neighbour[f"sigma_abs_shift_pb_{method}"])
            rows.append(
                {
                    "method": method,
                    "neighbour_rank": rank,
                    "ptcone_max": float(neighbour["ptcone_max"]),
                    "etcone_max": float(neighbour["etcone_max"]),
                    "delta_ptcone": float(neighbour["ptcone_max"] - nominal_ptcone),
                    "delta_etcone": float(neighbour["etcone_max"] - nominal_etcone),
                    "distance_to_nominal": distance,
                    "sigma_shift_pb": float(neighbour[f"sigma_shift_pb_{method}"]),
                    "sigma_abs_shift_pb": abs_shift,
                    "sigma_frac_shift": float(neighbour[f"sigma_frac_shift_{method}"]),
                    "abs_slope_pb_per_gev": abs_shift / distance if distance > 0 else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def monotonicity_summary_text(
    classification_table: pd.DataFrame,
    plateau_table: pd.DataFrame | None = None,
    methods: Sequence[str] | None = None,
) -> str:
    if classification_table.empty:
        return "No monotonicity slices were available."

    resolved_methods = tuple(method for method in (methods or DD_METHODS))
    lines = []
    for method in resolved_methods:
        method_rows = classification_table[classification_table["method"] == method]
        if method_rows.empty:
            continue
        lines.append(f"Method: {method}")
        counts = method_rows.groupby(["fixed_axis", "classification"]).size().reset_index(name="count")
        lines.append(counts.to_string(index=False))
        if plateau_table is not None and not plateau_table.empty:
            plateau_row = plateau_table[plateau_table["method"] == method]
            if not plateau_row.empty:
                lines.append(plateau_row[["plateau_assessment"]].to_string(index=False, header=False))
        lines.append("")
    return "\n".join(lines).strip()


def local_stability_summary_text(
    local_stability_table: pd.DataFrame,
    methods: Sequence[str] | None = None,
) -> str:
    if local_stability_table.empty:
        return "No neighbouring scan points were available around the nominal isolation point."

    resolved_methods = tuple(method for method in (methods or DD_METHODS))
    lines = []
    for method in resolved_methods:
        method_rows = local_stability_table[local_stability_table["method"] == method].sort_values("neighbour_rank")
        if method_rows.empty:
            continue
        nearest = method_rows.iloc[0]
        lines.append(
            (
                f"{method}: nearest point at "
                f"(ptcone={nearest['ptcone_max']:.3f}, etcone={nearest['etcone_max']:.3f}) gives "
                f"Δsigma={nearest['sigma_shift_pb']:.6f} pb, "
                f"|Δsigma|={nearest['sigma_abs_shift_pb']:.6f} pb, "
                f"fractional |Δsigma|={nearest['sigma_frac_shift']:.6f}."
            )
        )
    return "\n".join(lines)
