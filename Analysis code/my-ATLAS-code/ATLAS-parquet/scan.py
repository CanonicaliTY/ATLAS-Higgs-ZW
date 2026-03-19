from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from selections import apply_selection
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

        _, os_signal, _, os_signal_over_background, os_data_over_mc = summarize(os_selected)
        ss_data, _, _, _, _ = summarize(ss_selected)

        os_signal_efficiency = (os_signal / os_signal_reference) if os_signal_reference > 0 else float("nan")
        ss_rejection = 1.0 - (ss_data / ss_data_reference) if ss_data_reference > 0 else float("nan")

        rows.append(
            [
                ptcone_max,
                etcone_max,
                os_signal_efficiency,
                ss_rejection,
                os_data_over_mc,
                os_signal_over_background,
                ss_data,
            ]
        )

    full_scan = pd.DataFrame(
        rows,
        columns=["ptcone_max", "etcone_max", "OS_sig_eff", "SS_rejection", "OS_Data/MC", "OS_S/B", "SS_Data"],
    )
    allowed_scan = full_scan[full_scan["OS_sig_eff"] >= os_sig_eff_min].copy()
    allowed_scan["tightness"] = allowed_scan["ptcone_max"] + allowed_scan["etcone_max"]
    allowed_scan = allowed_scan.sort_values(["SS_rejection", "tightness"], ascending=[False, True])

    if allowed_scan.empty:
        raise RuntimeError(
            f"No scan point satisfies OS_sig_eff_min={os_sig_eff_min}. "
            "Try loosening ISO_SCAN ranges or lowering OS_SIG_EFF_MIN."
        )

    best_point = allowed_scan.iloc[0]
    return full_scan, allowed_scan, best_point


def scan_table_with_sigma(
    scan_table: pd.DataFrame,
    sigma_evaluator: Callable[[float, float], dict],
    method: str,
    order_mode: str,
) -> pd.DataFrame:
    rows = []
    for row in scan_table.to_dict(orient="records"):
        evaluation = sigma_evaluator(float(row["ptcone_max"]), float(row["etcone_max"]))
        sigma_result = evaluation["sigma_with_additional_bkg"]
        rows.append(
            {
                "ptcone_max": float(row["ptcone_max"]),
                "etcone_max": float(row["etcone_max"]),
                "OS_sig_eff": float(row["OS_sig_eff"]),
                "SS_rejection": float(row["SS_rejection"]),
                "OS_Data/MC": float(row["OS_Data/MC"]),
                "OS_S/B": float(row["OS_S/B"]),
                "SS_Data": float(row["SS_Data"]),
                "method": method,
                "order_mode": order_mode,
                "stage_name": evaluation["estimator_detail"]["stage_name"],
                "extra_bkg": evaluation["applied_extra_background"],
                "sigma_pb": sigma_result["sigma_pb"],
                "sigma_without_extra_pb": evaluation["sigma_without_additional_bkg"]["sigma_pb"],
            }
        )
    return pd.DataFrame(rows)


def find_scan_neighbours(
    scan_table: pd.DataFrame,
    nominal_ptcone: float,
    nominal_etcone: float,
    ptcone_step: float,
    etcone_step: float,
) -> pd.DataFrame:
    candidates = [
        (nominal_ptcone - ptcone_step, nominal_etcone),
        (nominal_ptcone + ptcone_step, nominal_etcone),
        (nominal_ptcone, nominal_etcone - etcone_step),
        (nominal_ptcone, nominal_etcone + etcone_step),
    ]

    neighbours = []
    for ptcone_value, etcone_value in candidates:
        if ptcone_value < 0 or etcone_value < 0:
            continue
        matches = scan_table[
            np.isclose(scan_table["ptcone_max"], ptcone_value) & np.isclose(scan_table["etcone_max"], etcone_value)
        ]
        if not matches.empty:
            neighbours.append(matches.iloc[0].to_dict())

    return pd.DataFrame(neighbours)
