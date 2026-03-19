from __future__ import annotations

import awkward as ak
import numpy as np
import vector

from config import CHANNELS, RAW_BUILD_VARS_COMMON, SETTINGS
from utils import weight_field


def add_mass(events: ak.Array) -> ak.Array:
    p4 = vector.zip(
        {
            "pt": events["lep_pt"],
            "eta": events["lep_eta"],
            "phi": events["lep_phi"],
            "E": events["lep_e"],
        }
    )
    return ak.with_field(events, (p4[:, 0] + p4[:, 1]).M, "mass")


def add_charge_product(events: ak.Array) -> ak.Array:
    charge_product = events["lep_charge"][:, 0] * events["lep_charge"][:, 1]
    return ak.with_field(events, charge_product, "charge_product")


def slim_keep_fields(events: ak.Array, keep_fields: list[str]) -> ak.Array:
    final_fields: list[str] = []
    for field in keep_fields:
        if field in events.fields and field not in final_fields:
            final_fields.append(field)
    event_weight_field = weight_field(events)
    if event_weight_field is not None and event_weight_field not in final_fields:
        final_fields.append(event_weight_field)
    return events[final_fields]


def main_build_keep_fields() -> list[str]:
    keep = list(SETTINGS["TIGHT_PARQUET"]["MAIN_KEEP_FIELDS"])
    for field in ["lep_pt", "lep_ptvarcone30", "lep_topoetcone20", "mass", "charge_product"]:
        if field not in keep:
            keep.append(field)
    return keep


def control_build_keep_fields() -> list[str]:
    keep = list(SETTINGS["TIGHT_PARQUET"]["CONTROL_KEEP_FIELDS"])
    for field in [
        "lep_pt",
        "lep_type",
        "lep_charge",
        "lep_ptvarcone30",
        "lep_topoetcone20",
        "mass",
        "charge_product",
    ]:
        if field not in keep:
            keep.append(field)
    return keep


def baseline_main_preselection(
    events: ak.Array,
    lepton: str,
    pt_min: float,
    apply_medium_id: bool,
    medium_field: str | None,
) -> ak.Array:
    channel = CHANNELS[lepton]
    selected = events[events["lep_n"] == 2]
    selected = selected[(selected["lep_type"][:, 0] + selected["lep_type"][:, 1]) == channel["type_sum"]]
    selected = selected[(selected["lep_pt"][:, 0] > pt_min) & (selected["lep_pt"][:, 1] > pt_min)]
    selected = selected[selected[channel["trigger_field"]]]
    if apply_medium_id and medium_field is not None and medium_field in selected.fields:
        selected = selected[selected[medium_field][:, 0] & selected[medium_field][:, 1]]
    selected = add_mass(selected)
    selected = add_charge_product(selected)
    return selected


def baseline_control_preselection(
    events: ak.Array,
    pt_min: float,
    trigger_mode: str,
    apply_medium_id: bool,
    medium_field: str | None,
) -> ak.Array:
    selected = events[events["lep_n"] == 2]

    type0 = selected["lep_type"][:, 0]
    type1 = selected["lep_type"][:, 1]
    is_e_or_mu = ((type0 == 11) | (type0 == 13)) & ((type1 == 11) | (type1 == 13))
    selected = selected[is_e_or_mu]
    selected = selected[(selected["lep_pt"][:, 0] > pt_min) & (selected["lep_pt"][:, 1] > pt_min)]

    trigger_mode = trigger_mode.lower().strip()
    if trigger_mode == "or":
        selected = selected[selected["trigE"] | selected["trigM"]]
    elif trigger_mode == "mu":
        selected = selected[selected["trigM"]]
    elif trigger_mode == "e":
        selected = selected[selected["trigE"]]
    else:
        raise ValueError(f"Unknown CONTROL_TRIGGER_MODE: {trigger_mode}")

    if apply_medium_id and medium_field is not None and medium_field in selected.fields:
        selected = selected[selected[medium_field][:, 0] & selected[medium_field][:, 1]]

    selected = add_mass(selected)
    selected = add_charge_product(selected)
    return selected


def make_main_build_cut(
    lepton: str,
    apply_medium_id: bool,
    medium_field: str | None,
    stage: str = "baseline",
    required_input_fields: list[str] | None = None,
):
    stage = stage.lower().strip()
    keep = main_build_keep_fields()
    for field in required_input_fields or []:
        if field not in keep:
            keep.append(field)

    def cut_function(events: ak.Array) -> ak.Array:
        selected = baseline_main_preselection(
            events,
            lepton=lepton,
            pt_min=SETTINGS["PT_MIN"],
            apply_medium_id=apply_medium_id,
            medium_field=medium_field,
        )
        if stage != "baseline":
            raise ValueError(f"Unsupported MAIN_STAGE={stage!r}; use 'baseline'.")
        return slim_keep_fields(selected, keep)

    return cut_function


def make_control_build_cut(
    apply_medium_id: bool,
    medium_field: str | None,
    required_input_fields: list[str] | None = None,
):
    keep = control_build_keep_fields()
    for field in required_input_fields or []:
        if field not in keep:
            keep.append(field)

    def cut_function(events: ak.Array) -> ak.Array:
        selected = baseline_control_preselection(
            events,
            pt_min=SETTINGS["PT_MIN"],
            trigger_mode=SETTINGS["ADDITIONAL_DATA_DRIVEN_BKG"]["CONTROL_TRIGGER_MODE"],
            apply_medium_id=apply_medium_id,
            medium_field=medium_field,
        )
        return slim_keep_fields(selected, keep)

    return cut_function


def make_sign_cut(sign: str):
    sign = sign.upper().strip()

    def cut_function(events: ak.Array) -> ak.Array:
        if "charge_product" not in events.fields:
            raise KeyError("charge_product not found in tight parquet; cannot split OS/SS")
        if sign == "OS":
            return events[events["charge_product"] < 0]
        if sign == "SS":
            return events[events["charge_product"] > 0]
        raise ValueError("sign must be 'OS' or 'SS'")

    return cut_function


def make_raw_sign_cut(lepton: str, sign: str):
    sign = sign.upper().strip()

    def cut_function(events: ak.Array) -> ak.Array:
        selected = baseline_main_preselection(
            events,
            lepton=lepton,
            pt_min=SETTINGS["PT_MIN"],
            apply_medium_id=False,
            medium_field=None,
        )
        if sign == "OS":
            return selected[selected["charge_product"] < 0]
        if sign == "SS":
            return selected[selected["charge_product"] > 0]
        raise ValueError("sign must be 'OS' or 'SS'")

    return cut_function


def apply_selection(
    events: ak.Array | None,
    mass_window: tuple[float, float] | None = None,
    ptcone_max: float | None = None,
    etcone_max: float | None = None,
    require_both: bool = True,
) -> ak.Array | None:
    if events is None:
        return None

    mask = ak.Array(np.ones(len(events), dtype=bool))

    if mass_window is not None:
        low_edge, high_edge = mass_window
        mask = mask & (events["mass"] > low_edge) & (events["mass"] < high_edge)

    if ptcone_max is not None:
        if require_both:
            mask = mask & (events["lep_ptvarcone30"][:, 0] < ptcone_max) & (events["lep_ptvarcone30"][:, 1] < ptcone_max)
        else:
            mask = mask & (events["lep_ptvarcone30"][:, 0] < ptcone_max)

    if etcone_max is not None:
        if require_both:
            mask = mask & (events["lep_topoetcone20"][:, 0] < etcone_max) & (events["lep_topoetcone20"][:, 1] < etcone_max)
        else:
            mask = mask & (events["lep_topoetcone20"][:, 0] < etcone_max)

    return events[mask]


def make_final_control_cut(ptcone_max: float, etcone_max: float, require_both: bool):
    def cut_function(events: ak.Array) -> ak.Array:
        return apply_selection(
            events,
            mass_window=SETTINGS["MASS_WINDOW"],
            ptcone_max=ptcone_max,
            etcone_max=etcone_max,
            require_both=require_both,
        )

    return cut_function


def raw_build_vars_common() -> list[str]:
    return list(RAW_BUILD_VARS_COMMON)

