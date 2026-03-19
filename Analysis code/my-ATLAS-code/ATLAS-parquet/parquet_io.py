from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable

import awkward as ak
import pandas as pd
import pyarrow.parquet as pq

from config import (
    CHANNELS,
    SCRIPT_DIR,
    SETTINGS,
    control_sample_codes_for_build,
    label_float,
    medium_id_mode,
)
from selections import (
    control_build_keep_fields,
    main_build_keep_fields,
    make_control_build_cut,
    make_main_build_cut,
    make_raw_sign_cut,
    make_sign_cut,
    raw_build_vars_common,
)
from utils import (
    infer_sample_code_from_name,
    is_data_sample,
    now_stamp,
    safe_rmtree,
    tight_fields_of_subdir,
    write_json,
)


def available_raw_fields(sample_code: str, backend: dict) -> list[str]:
    return list(backend["get_valid_variables"](sample_code))


def choose_medium_field(sample_code: str, backend: dict) -> str | None:
    fields = set(available_raw_fields(sample_code, backend))
    for candidate in SETTINGS["MEDIUM_ID"]["FIELD_CANDIDATES"]:
        if candidate in fields:
            return candidate
    return None


def should_apply_medium_id(sample_code: str, medium_field: str | None) -> tuple[bool, str]:
    if not SETTINGS["MEDIUM_ID"]["APPLY"]:
        return False, "disabled"
    if medium_field is None:
        return False, "field_not_available"

    scope = SETTINGS["MEDIUM_ID"]["SCOPE"]
    if scope == "disabled":
        return False, "disabled"
    if scope == "mc_only_if_available":
        if is_data_sample(sample_code):
            return False, "data_sample_skipped"
        return True, "mc_field_available"
    if scope == "if_available_all":
        return True, "field_available"
    return False, f"unknown_scope:{scope}"


def main_tight_tag(lepton: str) -> str:
    manual_tag = SETTINGS["TIGHT_PARQUET"]["MAIN_TAG"]
    if manual_tag:
        return str(manual_tag)
    return (
        f"{lepton}_main_{SETTINGS['TIGHT_PARQUET']['MAIN_STAGE']}"
        f"_pt{label_float(SETTINGS['PT_MIN'])}_mid_{medium_id_mode()}"
    )


def control_tight_tag() -> str:
    manual_tag = SETTINGS["TIGHT_PARQUET"]["CONTROL_TAG"]
    if manual_tag:
        return str(manual_tag)
    trigger_mode = SETTINGS["ADDITIONAL_DATA_DRIVEN_BKG"]["CONTROL_TRIGGER_MODE"]
    return f"control2lep_pt{label_float(SETTINGS['PT_MIN'])}_trig_{trigger_mode}_mid_{medium_id_mode()}"


def tight_root(kind: str, lepton: str | None = None) -> Path:
    base = (SCRIPT_DIR / SETTINGS["TIGHT_PARQUET"]["ROOT_DIR"]).resolve()
    if kind == "main":
        if lepton is None:
            raise ValueError("lepton is required for main tight parquet roots")
        return base / main_tight_tag(lepton)
    if kind == "control":
        return base / control_tight_tag()
    raise ValueError(f"Unknown tight-parquet kind: {kind}")


def manifest_path(root: Path) -> Path:
    return root / "_manifest.json"


def read_manifest(root: Path) -> dict | None:
    path = manifest_path(root)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_complete(root: Path) -> bool:
    manifest = read_manifest(root)
    return bool(manifest and manifest.get("complete"))


def reset_root_for_rebuild(root: Path) -> None:
    safe_rmtree(root)
    root.mkdir(parents=True, exist_ok=True)


def sample_output_subdirs(temp_root: Path) -> list[Path]:
    return [path for path in temp_root.iterdir() if path.is_dir() and not path.name.startswith("_")]


def build_one_sample_to_root(
    sample_code: str,
    root: Path,
    read_vars: list[str],
    cut_function,
    backend: dict,
    fraction: float,
) -> str | None:
    analysis_parquet = backend["analysis_parquet"]

    temp_root = root / f"_tmp_{sample_code}_{now_stamp()}"
    safe_rmtree(temp_root)
    temp_root.mkdir(parents=True, exist_ok=True)

    analysis_parquet(
        read_vars,
        [sample_code],
        fraction=fraction,
        cut_function=cut_function,
        write_parquet=True,
        output_directory=str(temp_root),
        return_output=False,
    )

    produced_subdirs = sample_output_subdirs(temp_root)
    if not produced_subdirs:
        safe_rmtree(temp_root)
        return None
    if len(produced_subdirs) != 1:
        raise RuntimeError(
            f"Expected exactly one written subdirectory for {sample_code}, found {[path.name for path in produced_subdirs]}"
        )

    source = produced_subdirs[0]
    destination = root / source.name
    if destination.exists():
        safe_rmtree(destination)
    shutil.move(str(source), str(destination))
    safe_rmtree(temp_root)
    return destination.name


def ensure_main_tight_parquet(lepton: str, backend: dict) -> Path:
    if not SETTINGS["USE_TIGHT_PARQUET"]:
        raise RuntimeError("ensure_main_tight_parquet called with USE_TIGHT_PARQUET=False")

    root = tight_root("main", lepton)
    if root.exists() and manifest_complete(root) and not SETTINGS["TIGHT_PARQUET"]["FORCE_REBUILD"]:
        return root

    reset_root_for_rebuild(root)

    build_fraction = SETTINGS["TIGHT_PARQUET"]["BUILD_FRACTION"]
    stage = SETTINGS["TIGHT_PARQUET"]["MAIN_STAGE"]
    sample_rows = []
    subdirs = []

    for sample_code in CHANNELS[lepton]["string_codes"]:
        raw_fields = set(available_raw_fields(sample_code, backend))
        medium_field = choose_medium_field(sample_code, backend)
        apply_medium_id, reason = should_apply_medium_id(sample_code, medium_field)

        needed_raw = [field for field in raw_build_vars_common() if field in raw_fields]
        if medium_field is not None and medium_field not in needed_raw:
            needed_raw.append(medium_field)

        cut_function = make_main_build_cut(
            lepton=lepton,
            apply_medium_id=apply_medium_id,
            medium_field=medium_field,
            stage=stage,
            required_input_fields=needed_raw,
        )
        output_subdir = build_one_sample_to_root(
            sample_code=sample_code,
            root=root,
            read_vars=needed_raw,
            cut_function=cut_function,
            backend=backend,
            fraction=build_fraction,
        )
        sample_rows.append(
            {
                "sample": sample_code,
                "output_subdir": output_subdir,
                "medium_id_field": medium_field,
                "apply_medium_id": apply_medium_id,
                "apply_medium_id_reason": reason,
                "read_vars": needed_raw,
            }
        )
        if output_subdir is not None:
            subdirs.append(output_subdir)

    manifest = {
        "complete": True,
        "kind": "main",
        "channel": lepton,
        "tag": main_tight_tag(lepton),
        "created_at": now_stamp(),
        "build_fraction": build_fraction,
        "main_stage": stage,
        "settings_snapshot": {
            "PT_MIN": SETTINGS["PT_MIN"],
            "MEDIUM_ID": SETTINGS["MEDIUM_ID"],
            "MAIN_KEEP_FIELDS": main_build_keep_fields(),
        },
        "subdirs": subdirs,
        "samples": sample_rows,
    }
    if SETTINGS["TIGHT_PARQUET"]["WRITE_METADATA"]:
        write_json(manifest_path(root), manifest)
        pd.DataFrame(sample_rows).to_csv(root / "_medium_id_usage.csv", index=False)
    return root


def ensure_control_tight_parquet(backend: dict) -> Path:
    root = tight_root("control")
    if root.exists() and manifest_complete(root) and not SETTINGS["ADDITIONAL_DATA_DRIVEN_BKG"]["FORCE_REBUILD"]:
        return root

    reset_root_for_rebuild(root)

    build_fraction = SETTINGS["TIGHT_PARQUET"]["BUILD_FRACTION"]
    sample_rows = []
    subdirs = []

    for sample_code in control_sample_codes_for_build():
        raw_fields = set(available_raw_fields(sample_code, backend))
        medium_field = choose_medium_field(sample_code, backend)
        apply_medium_id, reason = should_apply_medium_id(sample_code, medium_field)

        needed_raw = [field for field in raw_build_vars_common() if field in raw_fields]
        if medium_field is not None and medium_field not in needed_raw:
            needed_raw.append(medium_field)

        cut_function = make_control_build_cut(
            apply_medium_id=apply_medium_id,
            medium_field=medium_field,
            required_input_fields=needed_raw,
        )
        output_subdir = build_one_sample_to_root(
            sample_code=sample_code,
            root=root,
            read_vars=needed_raw,
            cut_function=cut_function,
            backend=backend,
            fraction=build_fraction,
        )
        sample_rows.append(
            {
                "sample": sample_code,
                "output_subdir": output_subdir,
                "medium_id_field": medium_field,
                "apply_medium_id": apply_medium_id,
                "apply_medium_id_reason": reason,
                "read_vars": needed_raw,
            }
        )
        if output_subdir is not None:
            subdirs.append(output_subdir)

    manifest = {
        "complete": True,
        "kind": "control",
        "tag": control_tight_tag(),
        "created_at": now_stamp(),
        "build_fraction": build_fraction,
        "settings_snapshot": {
            "PT_MIN": SETTINGS["PT_MIN"],
            "MEDIUM_ID": SETTINGS["MEDIUM_ID"],
            "CONTROL_TRIGGER_MODE": SETTINGS["ADDITIONAL_DATA_DRIVEN_BKG"]["CONTROL_TRIGGER_MODE"],
            "CONTROL_KEEP_FIELDS": control_build_keep_fields(),
        },
        "subdirs": subdirs,
        "samples": sample_rows,
    }
    if SETTINGS["TIGHT_PARQUET"]["WRITE_METADATA"]:
        write_json(manifest_path(root), manifest)
        pd.DataFrame(sample_rows).to_csv(root / "_medium_id_usage.csv", index=False)
    return root


def required_main_read_fields() -> list[str]:
    return ["lep_pt", "lep_ptvarcone30", "lep_topoetcone20", "mass", "charge_product", "weight", "totalWeight"]


def required_control_read_fields() -> list[str]:
    return [
        "lep_pt",
        "lep_type",
        "lep_charge",
        "lep_ptvarcone30",
        "lep_topoetcone20",
        "mass",
        "charge_product",
        "weight",
        "totalWeight",
    ]


def load_tight_subdirs(
    root: Path,
    subdirs: list[str],
    needed_fields: list[str],
    backend: dict,
    fraction: float,
    cut_function=None,
) -> dict:
    analysis_parquet = backend["analysis_parquet"]
    loaded = {}
    for subdir_name in subdirs:
        subdir = root / subdir_name
        available_fields = set(tight_fields_of_subdir(subdir))
        read_vars = [field for field in needed_fields if field in available_fields]
        chunk = analysis_parquet(
            read_variables=read_vars,
            string_code_list=None,
            read_directory=str(root),
            subdirectory_names=[subdir_name],
            fraction=fraction,
            cut_function=cut_function,
            write_parquet=False,
            output_directory=None,
            return_output=True,
        )
        loaded.update(chunk)
    return loaded


def load_main_events(lepton: str, sign: str, backend: dict) -> dict:
    if SETTINGS["USE_TIGHT_PARQUET"]:
        root = ensure_main_tight_parquet(lepton, backend)
        manifest = read_manifest(root)
        if manifest is None:
            raise RuntimeError(f"Missing manifest under {root}")
        return load_tight_subdirs(
            root=root,
            subdirs=manifest["subdirs"],
            needed_fields=required_main_read_fields(),
            backend=backend,
            fraction=SETTINGS["FRACTION"],
            cut_function=make_sign_cut(sign),
        )

    validate_read_variables = backend["validate_read_variables"]
    analysis_parquet = backend["analysis_parquet"]
    read_vars = validate_read_variables(CHANNELS[lepton]["string_codes"], raw_build_vars_common())
    return analysis_parquet(
        read_vars,
        CHANNELS[lepton]["string_codes"],
        fraction=SETTINGS["FRACTION"],
        cut_function=make_raw_sign_cut(lepton, sign),
    )


def collect_channel_control_samples(lepton: str) -> dict:
    data_sample = "2to4lep"
    mc_samples = [sample for sample in control_sample_codes_for_build() if not is_data_sample(sample)]
    all_samples = [data_sample] + [sample for sample in mc_samples if sample != data_sample]
    return {"data_sample": data_sample, "mc_samples": mc_samples, "all_samples": all_samples}


def _chunk_read_columns_for_control(file_fields: set[str]) -> list[str]:
    required = [
        "lep_pt",
        "lep_type",
        "lep_charge",
        "lep_ptvarcone30",
        "lep_topoetcone20",
        "mass",
    ]
    if "charge_product" in file_fields:
        required.append("charge_product")

    optional = [field for field in ("weight", "totalWeight") if field in file_fields]
    columns = [field for field in required if field in file_fields]
    for field in optional:
        if field not in columns:
            columns.append(field)
    return columns


def accumulate_control_stage_totals_from_tight_chunks(
    backend: dict,
    lepton: str,
    stage_selectors: dict[str, Callable[[ak.Array], ak.Array | None]],
    stage_region_summariser,
    region_names: list[str],
) -> dict:
    if not SETTINGS["ADDITIONAL_DATA_DRIVEN_BKG"]["USE_CONTROL_TIGHT_PARQUET"]:
        raise RuntimeError(
            "Chunk-level control accumulation requires USE_CONTROL_TIGHT_PARQUET=True. "
            "Disable the additional background study or enable the control tight parquet."
        )

    root = ensure_control_tight_parquet(backend)
    manifest = read_manifest(root)
    if manifest is None:
        raise RuntimeError(f"Missing control manifest under {root}")

    sample_info = collect_channel_control_samples(lepton)
    data_sample = sample_info["data_sample"]
    mc_sample_set = set(sample_info["mc_samples"])
    relevant_samples = set(sample_info["all_samples"])

    subdir_to_sample = {}
    for row in manifest.get("samples", []):
        if row.get("output_subdir") and row.get("sample"):
            subdir_to_sample[str(row["output_subdir"])] = str(row["sample"])

    totals = {
        stage_name: {
            "data_counts": {region: 0.0 for region in region_names},
            "mc_counts": {region: 0.0 for region in region_names},
            "mc_samples_included": set(),
        }
        for stage_name in stage_selectors
    }

    for subdir_name in manifest["subdirs"]:
        sample_code = subdir_to_sample.get(subdir_name)
        if sample_code is None:
            sample_code = infer_sample_code_from_name(subdir_name, list(relevant_samples))
        if sample_code is None or sample_code not in relevant_samples:
            continue

        subdir = root / subdir_name
        for parquet_file in sorted(subdir.rglob("*.parquet")):
            parquet_handle = pq.ParquetFile(parquet_file)
            file_fields = set(parquet_handle.schema_arrow.names)
            missing = {"lep_type", "lep_charge", "lep_ptvarcone30", "lep_topoetcone20", "mass"} - file_fields
            if missing:
                raise KeyError(f"Missing required control fields in {parquet_file}: {sorted(missing)}")

            columns = _chunk_read_columns_for_control(file_fields)
            for row_group in range(parquet_handle.num_row_groups):
                table = parquet_handle.read_row_group(row_group, columns=columns)
                if table.num_rows == 0:
                    continue

                chunk = ak.from_arrow(table)
                if "charge_product" not in chunk.fields:
                    chunk = ak.with_field(chunk, chunk["lep_charge"][:, 0] * chunk["lep_charge"][:, 1], "charge_product")

                for stage_name, selector in stage_selectors.items():
                    selected = selector(chunk)
                    summary = stage_region_summariser(selected)
                    if sample_code == data_sample:
                        for region in region_names:
                            totals[stage_name]["data_counts"][region] += summary["count"][region]
                    elif sample_code in mc_sample_set:
                        for region in region_names:
                            totals[stage_name]["mc_counts"][region] += summary["weight"][region]
                        totals[stage_name]["mc_samples_included"].add(sample_code)

    for stage_name in totals:
        totals[stage_name]["mc_samples_included"] = sorted(totals[stage_name]["mc_samples_included"])

    return totals
