from __future__ import annotations

import math

from config import LUMI_FB, LUMI_PB
from utils import produced_sumw, yield_data, yield_mc, yield_mc_var
from visualisation import select_plotdict


def compute_sigma(
    plot_os: dict,
    channel_config: dict,
    produced_event_count_fn,
    mass_window: tuple[float, float],
    ptcone_max: float,
    etcone_max: float,
    require_both: bool,
    extra_bkg: float = 0.0,
    produced_sumw_cache: dict[str, float] | None = None,
) -> dict:
    primary_signal = channel_config["primary_signal"]
    primary_signal_label = f"Signal {primary_signal}"

    selected = select_plotdict(
        plot_os,
        mass_window=mass_window,
        ptcone_max=ptcone_max,
        etcone_max=etcone_max,
        require_both=require_both,
    )

    n_selected = yield_data(selected.get("Data"))

    n_background_mc = 0.0
    n_background_var = 0.0
    for label, events in selected.items():
        if label in {"Data", primary_signal_label}:
            continue
        n_background_mc += yield_mc(events)
        n_background_var += yield_mc_var(events)

    n_background_total = n_background_mc + float(extra_bkg)
    n_signal_data = n_selected - n_background_total

    signal_pass_weight = yield_mc(selected.get(primary_signal_label))
    signal_total_weight = produced_sumw(
        produced_event_count_fn,
        primary_signal,
        LUMI_FB,
        cache=produced_sumw_cache,
    )
    if signal_total_weight <= 0:
        raise ZeroDivisionError(f"Produced sum of weights is non-positive for primary signal {primary_signal!r}")

    efficiency = signal_pass_weight / signal_total_weight
    if efficiency <= 0:
        raise ZeroDivisionError(f"Efficiency is non-positive for primary signal {primary_signal!r}")

    sigma_pb = n_signal_data / (efficiency * LUMI_PB)

    d_n_data = math.sqrt(max(n_selected, 0.0))
    d_n_background = math.sqrt(max(n_background_var, 0.0))
    d_n_signal = math.sqrt(d_n_data**2 + d_n_background**2)
    d_sigma_stat_pb = d_n_signal / (efficiency * LUMI_PB)

    return {
        "primary_signal": primary_signal,
        "N_selected": n_selected,
        "N_bkg_mc": n_background_mc,
        "N_bkg_extra": float(extra_bkg),
        "N_bkg_total": n_background_total,
        "N_sig_data": n_signal_data,
        "epsilon": efficiency,
        "sigma_pb": sigma_pb,
        "dsigma_stat_pb": d_sigma_stat_pb,
    }

