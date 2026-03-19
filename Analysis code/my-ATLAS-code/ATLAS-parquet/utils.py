from __future__ import annotations

import contextlib
import datetime
import io
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import awkward as ak
import pyarrow.parquet as pq

from config import SCRIPT_DIR, SETTINGS


def ensure_script_directory() -> None:
    os.chdir(SCRIPT_DIR)


def now_stamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def write_json(path: Path, obj: Any) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=False)


def write_text(path: Path, text: str) -> None:
    ensure_parent(path)
    path.write_text(text, encoding="utf-8")


def _pip_install(requirement: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", requirement])


def ensure_environment() -> None:
    if not SETTINGS["AUTO_INSTALL"] and not SETTINGS["RUN_INSTALL_FROM_ENV_YML"]:
        return

    try:
        import atlasopenmagic  # noqa: F401
    except Exception:
        if SETTINGS["AUTO_INSTALL"]:
            _pip_install("atlasopenmagic")
        else:
            raise

    try:
        import pyarrow  # noqa: F401
        if SETTINGS["AUTO_INSTALL"] and getattr(pyarrow, "__version__", "") != "20.0.0":
            _pip_install("pyarrow==20.0.0")
    except Exception:
        if SETTINGS["AUTO_INSTALL"]:
            _pip_install("pyarrow==20.0.0")
        else:
            raise

    if SETTINGS["RUN_INSTALL_FROM_ENV_YML"]:
        from atlasopenmagic import install_from_environment

        env_file = (SCRIPT_DIR / "../backend/environment.yml").resolve()
        install_from_environment(environment_file=str(env_file))


def import_backend() -> dict[str, Any]:
    parent = SCRIPT_DIR.parent
    if str(parent) not in sys.path:
        sys.path.append(str(parent))

    from backend import (
        analysis_parquet,
        get_valid_variables,
        plot_stacked_hist,
        produced_event_count,
        validate_read_variables,
    )

    return {
        "analysis_parquet": analysis_parquet,
        "get_valid_variables": get_valid_variables,
        "plot_stacked_hist": plot_stacked_hist,
        "produced_event_count": produced_event_count,
        "validate_read_variables": validate_read_variables,
    }


def is_data_sample(sample_code: str) -> bool:
    return sample_code in {"2to4lep", "GamGam"}


def get_sample_key_by_prefix(data_dict: dict[str, Any], sample_code: str) -> str | None:
    matches = [key for key in data_dict if key.startswith(sample_code + "_")]
    if not matches:
        return None
    return sorted(matches, key=len)[0]


def infer_sample_code_from_name(name: str, sample_codes: list[str]) -> str | None:
    for sample_code in sorted(sample_codes, key=len, reverse=True):
        if name == sample_code or name.startswith(sample_code + "_"):
            return sample_code
    return None


def tight_first_parquet_file(subdir: Path) -> Path:
    files = sorted(subdir.glob("*.parquet"))
    if files:
        return files[0]
    files = sorted(subdir.rglob("*.parquet"))
    if files:
        return files[0]
    raise FileNotFoundError(f"No parquet file found under {subdir}")


def tight_fields_of_subdir(subdir: Path) -> list[str]:
    return list(pq.read_table(tight_first_parquet_file(subdir)).schema.names)


def weight_field(events: ak.Array | None) -> str | None:
    if events is None:
        return None
    for candidate in ("weight", "totalWeight"):
        if candidate in events.fields:
            return candidate
    return None


def yield_data(events: ak.Array | None) -> float:
    return 0.0 if events is None else float(len(events))


def yield_mc(events: ak.Array | None) -> float:
    if events is None:
        return 0.0
    field = weight_field(events)
    if field is None:
        return float(len(events))
    return float(ak.sum(events[field]))


def yield_mc_var(events: ak.Array | None) -> float:
    if events is None:
        return 0.0
    field = weight_field(events)
    if field is None:
        return float(len(events))
    weights = events[field]
    return float(ak.sum(weights * weights))


def produced_sumw(
    produced_event_count_fn,
    sample_key: str,
    lumi_fb: float,
    cache: dict[str, float] | None = None,
) -> float:
    cache_key = f"{sample_key}:{lumi_fb}"
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        produced_event_count_fn(sample_key, lumi_fb)
    text = buffer.getvalue().strip()
    match = re.search(r"([0-9]+)\s*$", text)
    if not match:
        raise RuntimeError(f"Could not parse produced_event_count output:\n{text}")
    value = float(match.group(1))
    if cache is not None:
        cache[cache_key] = value
    return value

