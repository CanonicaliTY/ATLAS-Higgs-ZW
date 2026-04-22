from __future__ import annotations

"""
Focused additional-research workflow for the muon-channel FSR / bremsstrahlung study.

This script is intentionally separate from the main analysis pipeline so that the core lab
workflow stays stable and this additional study remains readable.

What it does
------------
1. Build (or reuse) a dedicated tight parquet that keeps the 4-vector information needed for
   toy FSR-style mass recovery.
2. Stream that parquet sample-by-sample and row-group-by-row-group.
3. Construct the control sample "fail nominal isolation but pass loose isolation".
4. Accumulate stacked mass histograms, mass-window tables, and sigma-scan totals without
   materialising all OS events in memory.

Important physics note
----------------------
This is a toy study, not a precision photon reconstruction. The correction interprets
`lep_topoetcone20` as a near-muon radiated-photon proxy and adds it back collinearly to the
muon 4-vector. It is therefore best viewed as a hypothesis test.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from config import CHANNELS, LUMI_FB, SCRIPT_DIR, SETTINGS
from cross_section import compute_sigma_from_totals
from parquet_io import (
    available_raw_fields,
    build_one_sample_to_root,
    choose_medium_field,
    manifest_complete,
    read_manifest,
    reset_root_for_rebuild,
    should_apply_medium_id,
)
from scan import generate_scan_points
from selections import baseline_main_preselection, slim_keep_fields
from utils import (
    ensure_environment,
    ensure_script_directory,
    infer_sample_code_from_name,
    import_backend,
    log_step,
    now_stamp,
    produced_sumw,
    progress_iter,
    weight_field,
    write_json,
    write_text,
)
from visualisation import save_fig, save_scan_heatmap


FSR_SETTINGS = {
    # This dedicated study is explicitly muon-channel only.
    "LEPTON": "mu",

    # Fractions for parquet build / readback.
    "BUILD_FRACTION": 1.0,
    "READ_FRACTION": 1.0,

    # Dedicated parquet for the FSR study.
    "FORCE_REBUILD": False,
    "ROOT_DIR": "../../tight-parquet-fsr",

    # Baseline dimuon preselection.
    "PT_MIN": float(SETTINGS["PT_MIN"]),
    "APPLY_MEDIUM_ID": bool(SETTINGS["MEDIUM_ID"]["APPLY"]),

    # Isolation working points for the control study.
    # These defaults reproduce the older working point requested in the task.
    "NOMINAL_ISO": {"ptcone_max": 9.0, "etcone_max": 6.0},
    "LOOSE_ISO": {"ptcone_max": 15.0, "etcone_max": 15.0},
    "REQUIRE_BOTH_ISO": True,

    # Signal-region mass window used for the sigma study.
    "MASS_WINDOW": tuple(SETTINGS["MASS_WINDOW"]),

    # Toy FSR proxy model.
    "FSR": {
        "APPLY_BELOW_MASS": 80.0,
        "ETCONE_SCALE": 1.0,
        "MODES": ("maxcone", "both"),
    },

    # Sigma scan around the nominal point (MC-only background subtraction).
    "SCAN": {
        "RUN": True,
        "MODE": "local_box",
        "PTCONE_RANGE": (0.0, 10.0),
        "PTCONE_STEP": 1.0,
        "ETCONE_RANGE": (0.0, 20.0),
        "ETCONE_STEP": 1.0,
        "LOCAL_BOX_PTCONE_HALF_WIDTH": 2.0,
        "LOCAL_BOX_ETCONE_HALF_WIDTH": 2.0,
    },

    # Plots / outputs.
    "OUTPUT_DIR": "output_fsr",
    "SAVE_PLOTS": True,
    "MASS_PLOT": {
        "FULL": {"xmin": 0.0, "xmax": 200.0, "bins": 120, "logy": True},
        "ZOOM": {"xmin": 40.0, "xmax": 120.0, "bins": 80, "logy": True},
    },
    "WINDOWS": [(10.0, 40.0), (40.0, 66.0), (66.0, 80.0), (80.0, 100.0), (100.0, 116.0)],
}


@dataclass(frozen=True)
class IsoWorkingPoint:
    ptcone_max: float
    etcone_max: float


@dataclass(frozen=True)
class PlotConfig:
    xmin: float
    xmax: float
    bins: int
    logy: bool


@dataclass(frozen=True)
class ScanConfig:
    run: bool
    mode: str
    ptcone_range: tuple[float, float]
    ptcone_step: float
    etcone_range: tuple[float, float]
    etcone_step: float
    local_box_ptcone_half_width: float
    local_box_etcone_half_width: float


@dataclass(frozen=True)
class FSRStudyConfig:
    lepton: str
    build_fraction: float
    read_fraction: float
    force_rebuild: bool
    root_dir: str
    pt_min: float
    apply_medium_id: bool
    nominal_iso: IsoWorkingPoint
    loose_iso: IsoWorkingPoint
    require_both_iso: bool
    mass_window: tuple[float, float]
    fsr_apply_below_mass: float
    fsr_etcone_scale: float
    fsr_modes: tuple[str, ...]
    scan: ScanConfig
    output_dir: str
    save_plots: bool
    mass_plots: dict[str, PlotConfig]
    windows: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class FSRSample:
    sample_code: str
    subdir_name: str
    subdir_path: Path
    plot_label: str
    yield_category: str
    is_primary_signal: bool


@dataclass(frozen=True)
class PlotSpec:
    selection_name: str
    mass_field: str
    plot_config: PlotConfig
    title: str


@dataclass(frozen=True)
class ScanStudySpec:
    mass_field: str
    etcone_metric_key: str
    title: str


@dataclass(frozen=True)
class ScanGrid:
    points: tuple[tuple[float, float], ...]
    pt_thresholds: tuple[float, ...]
    et_thresholds: tuple[float, ...]


@dataclass
class FSRChunkView:
    mass_raw: np.ndarray
    mass_fsr_maxcone: np.ndarray
    mass_fsr_both: np.ndarray
    ptcone_metric: np.ndarray
    etcone_metric_raw: np.ndarray
    etcone_metric_fsrsub_maxcone: np.ndarray
    weights: np.ndarray


@dataclass
class YieldSummary:
    data: float = 0.0
    signal: float = 0.0
    background: float = 0.0

    def add(self, sample: FSRSample, mask: np.ndarray, weights: np.ndarray) -> None:
        if mask.size == 0 or not np.any(mask):
            return

        if sample.yield_category == "data":
            self.data += float(np.count_nonzero(mask))
            return

        selected_weights = weights[mask]
        if sample.yield_category == "signal":
            self.signal += float(np.sum(selected_weights))
            return

        self.background += float(np.sum(selected_weights))

    def as_dict(self) -> dict[str, float]:
        total_mc = self.signal + self.background
        return {
            "Data": float(self.data),
            "Signal": float(self.signal),
            "Background": float(self.background),
            "Data_minus_MC": float(self.data - total_mc),
        }


@dataclass
class HistogramAccumulator:
    edges: np.ndarray
    counts_by_label: dict[str, np.ndarray]
    variances_by_label: dict[str, np.ndarray]

    @classmethod
    def from_plot_config(cls, plot_config: PlotConfig, labels: list[str]) -> "HistogramAccumulator":
        edges = np.linspace(plot_config.xmin, plot_config.xmax, plot_config.bins + 1, dtype=float)
        return cls(
            edges=edges,
            counts_by_label={label: np.zeros(plot_config.bins, dtype=float) for label in labels},
            variances_by_label={label: np.zeros(plot_config.bins, dtype=float) for label in labels},
        )

    def fill(self, label: str, values: np.ndarray, weights: np.ndarray) -> None:
        if values.size == 0:
            return

        self.counts_by_label[label] += np.histogram(values, bins=self.edges, weights=weights)[0]
        self.variances_by_label[label] += np.histogram(values, bins=self.edges, weights=np.square(weights))[0]


@dataclass
class SigmaStudyAccumulator:
    spec: ScanStudySpec
    data_counts: np.ndarray
    background_weights: np.ndarray
    background_vars: np.ndarray
    signal_weights: np.ndarray

    @classmethod
    def create(cls, spec: ScanStudySpec, n_points: int) -> "SigmaStudyAccumulator":
        zeros = np.zeros(n_points, dtype=float)
        return cls(
            spec=spec,
            data_counts=zeros.copy(),
            background_weights=zeros.copy(),
            background_vars=zeros.copy(),
            signal_weights=zeros.copy(),
        )


WINDOW_MASS_FIELDS = ("mass_raw", "mass_fsr_maxcone", "mass_fsr_both")
SCAN_STUDIES = {
    "raw": ScanStudySpec(
        mass_field="mass_raw",
        etcone_metric_key="etcone_metric_raw",
        title="sigma (raw mass, raw etcone)",
    ),
    "masscorr_maxcone": ScanStudySpec(
        mass_field="mass_fsr_maxcone",
        etcone_metric_key="etcone_metric_raw",
        title="sigma (mass corrected: maxcone)",
    ),
    "masscorr_both": ScanStudySpec(
        mass_field="mass_fsr_both",
        etcone_metric_key="etcone_metric_raw",
        title="sigma (mass corrected: both)",
    ),
    "masscorr_etsub_maxcone": ScanStudySpec(
        mass_field="mass_fsr_maxcone",
        etcone_metric_key="etcone_metric_fsrsub_maxcone",
        title="sigma (mass corrected + etcone-subtracted: maxcone)",
    ),
}


def resolve_fsr_config() -> FSRStudyConfig:
    lepton = str(FSR_SETTINGS["LEPTON"]).strip().lower()
    if lepton != "mu":
        raise ValueError("This focused FSR study is currently implemented for the muon channel only.")

    fsr_modes = tuple(str(mode) for mode in FSR_SETTINGS["FSR"]["MODES"])
    if set(fsr_modes) != {"maxcone", "both"}:
        raise ValueError("FSR_SETTINGS['FSR']['MODES'] must contain exactly ('maxcone', 'both') for this workflow.")

    return FSRStudyConfig(
        lepton=lepton,
        build_fraction=float(FSR_SETTINGS["BUILD_FRACTION"]),
        read_fraction=float(FSR_SETTINGS["READ_FRACTION"]),
        force_rebuild=bool(FSR_SETTINGS["FORCE_REBUILD"]),
        root_dir=str(FSR_SETTINGS["ROOT_DIR"]),
        pt_min=float(FSR_SETTINGS["PT_MIN"]),
        apply_medium_id=bool(FSR_SETTINGS["APPLY_MEDIUM_ID"]),
        nominal_iso=IsoWorkingPoint(
            ptcone_max=float(FSR_SETTINGS["NOMINAL_ISO"]["ptcone_max"]),
            etcone_max=float(FSR_SETTINGS["NOMINAL_ISO"]["etcone_max"]),
        ),
        loose_iso=IsoWorkingPoint(
            ptcone_max=float(FSR_SETTINGS["LOOSE_ISO"]["ptcone_max"]),
            etcone_max=float(FSR_SETTINGS["LOOSE_ISO"]["etcone_max"]),
        ),
        require_both_iso=bool(FSR_SETTINGS["REQUIRE_BOTH_ISO"]),
        mass_window=tuple(float(edge) for edge in FSR_SETTINGS["MASS_WINDOW"]),
        fsr_apply_below_mass=float(FSR_SETTINGS["FSR"]["APPLY_BELOW_MASS"]),
        fsr_etcone_scale=float(FSR_SETTINGS["FSR"]["ETCONE_SCALE"]),
        fsr_modes=fsr_modes,
        scan=ScanConfig(
            run=bool(FSR_SETTINGS["SCAN"]["RUN"]),
            mode=str(FSR_SETTINGS["SCAN"]["MODE"]),
            ptcone_range=tuple(float(edge) for edge in FSR_SETTINGS["SCAN"]["PTCONE_RANGE"]),
            ptcone_step=float(FSR_SETTINGS["SCAN"]["PTCONE_STEP"]),
            etcone_range=tuple(float(edge) for edge in FSR_SETTINGS["SCAN"]["ETCONE_RANGE"]),
            etcone_step=float(FSR_SETTINGS["SCAN"]["ETCONE_STEP"]),
            local_box_ptcone_half_width=float(FSR_SETTINGS["SCAN"]["LOCAL_BOX_PTCONE_HALF_WIDTH"]),
            local_box_etcone_half_width=float(FSR_SETTINGS["SCAN"]["LOCAL_BOX_ETCONE_HALF_WIDTH"]),
        ),
        output_dir=str(FSR_SETTINGS["OUTPUT_DIR"]),
        save_plots=bool(FSR_SETTINGS["SAVE_PLOTS"]),
        mass_plots={
            name: PlotConfig(
                xmin=float(plot_cfg["xmin"]),
                xmax=float(plot_cfg["xmax"]),
                bins=int(plot_cfg["bins"]),
                logy=bool(plot_cfg["logy"]),
            )
            for name, plot_cfg in FSR_SETTINGS["MASS_PLOT"].items()
        },
        windows=tuple(tuple(float(edge) for edge in window) for window in FSR_SETTINGS["WINDOWS"]),
    )


def _fsr_keep_fields() -> list[str]:
    return [
        "lep_pt",
        "lep_eta",
        "lep_phi",
        "lep_e",
        "lep_type",
        "lep_charge",
        "lep_ptvarcone30",
        "lep_topoetcone20",
        "mass",
        "charge_product",
    ]


def _fsr_root(config: FSRStudyConfig) -> Path:
    base = (SCRIPT_DIR / config.root_dir).resolve()
    tag = (
        f"{config.lepton}_fsrstudy"
        f"_pt{config.pt_min:.1f}"
        f"_mid_{'on' if config.apply_medium_id else 'off'}"
    ).replace(".", "p")
    return base / tag


def _fsr_manifest_path(root: Path) -> Path:
    return root / "_manifest.json"


def _fsr_build_cut(
    *,
    config: FSRStudyConfig,
    lepton: str,
    apply_medium_id: bool,
    medium_field: str | None,
    required_input_fields: list[str] | None = None,
):
    # Keep the final physics fields and the raw fields still needed by
    # baseline_main_preselection during the parquet-writing stage.
    keep = _fsr_keep_fields()
    for field in required_input_fields or []:
        if field not in keep:
            keep.append(field)

    def cut_function(events: ak.Array) -> ak.Array:
        selected = baseline_main_preselection(
            events,
            lepton=lepton,
            pt_min=config.pt_min,
            apply_medium_id=apply_medium_id,
            medium_field=medium_field,
        )
        return slim_keep_fields(selected, keep)

    return cut_function


def ensure_fsr_tight_parquet(config: FSRStudyConfig, backend: dict) -> Path:
    root = _fsr_root(config)
    if root.exists() and manifest_complete(root) and not config.force_rebuild:
        log_step(f"[{config.lepton}] Reusing FSR tight parquet")
        return root

    log_step(f"[{config.lepton}] Building FSR tight parquet")
    reset_root_for_rebuild(root)

    sample_rows: list[dict] = []
    subdirs: list[str] = []

    sample_codes = CHANNELS[config.lepton]["string_codes"]
    iterator = progress_iter(sample_codes, total=len(sample_codes), desc=f"{config.lepton} fsr parquet", unit="sample")
    for sample_code in iterator:
        raw_fields = set(available_raw_fields(sample_code, backend))
        medium_field = choose_medium_field(sample_code, backend)
        apply_medium_id, reason = should_apply_medium_id(sample_code, medium_field)
        if not config.apply_medium_id:
            apply_medium_id = False
            reason = "disabled_in_fsr_settings"

        needed_raw = [
            field
            for field in [
                "lep_n",
                "lep_pt",
                "lep_eta",
                "lep_phi",
                "lep_e",
                "lep_type",
                "lep_charge",
                "trigM",
                "trigE",
                "lep_ptvarcone30",
                "lep_topoetcone20",
            ]
            if field in raw_fields
        ]
        if medium_field is not None and medium_field not in needed_raw:
            needed_raw.append(medium_field)

        output_subdir = build_one_sample_to_root(
            sample_code=sample_code,
            root=root,
            read_vars=needed_raw,
            cut_function=_fsr_build_cut(
                config=config,
                lepton=config.lepton,
                apply_medium_id=apply_medium_id,
                medium_field=medium_field,
                required_input_fields=needed_raw,
            ),
            backend=backend,
            fraction=config.build_fraction,
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
        "kind": "fsr_main",
        "channel": config.lepton,
        "created_at": now_stamp(),
        "settings": {
            "PT_MIN": config.pt_min,
            "BUILD_FRACTION": config.build_fraction,
            "APPLY_MEDIUM_ID": config.apply_medium_id,
            "KEEP_FIELDS": _fsr_keep_fields(),
        },
        "subdirs": subdirs,
        "samples": sample_rows,
    }
    write_json(_fsr_manifest_path(root), manifest)
    return root


def _plot_label_for_sample(sample_code: str, lepton: str) -> str:
    channel = CHANNELS[lepton]
    if sample_code == "2to4lep":
        return "Data"
    if sample_code == channel["primary_signal"]:
        return f"Signal {sample_code}"
    return f"Background {sample_code}"


def _yield_category_for_sample(sample_code: str, lepton: str) -> str:
    if sample_code == "2to4lep":
        return "data"
    if sample_code == CHANNELS[lepton]["primary_signal"]:
        return "signal"
    return "background"


def ordered_plot_labels(lepton: str) -> list[str]:
    channel = CHANNELS[lepton]
    labels = ["Data", f"Signal {channel['primary_signal']}"]
    labels.extend(f"Background {sample_code}" for sample_code in channel["signal_samples"] if sample_code != channel["primary_signal"])
    labels.extend(f"Background {sample_code}" for sample_code in channel["background_samples"])
    return labels


def resolve_fsr_samples(root: Path, manifest: dict, lepton: str) -> list[FSRSample]:
    relevant_samples = list(CHANNELS[lepton]["string_codes"])
    subdir_to_sample = {}
    for row in manifest.get("samples", []):
        if row.get("output_subdir") and row.get("sample"):
            subdir_to_sample[str(row["output_subdir"])] = str(row["sample"])

    samples: list[FSRSample] = []
    for subdir_name in manifest.get("subdirs", []):
        sample_code = subdir_to_sample.get(subdir_name)
        if sample_code is None:
            sample_code = infer_sample_code_from_name(subdir_name, relevant_samples)
        if sample_code is None or sample_code not in relevant_samples:
            continue
        samples.append(
            FSRSample(
                sample_code=sample_code,
                subdir_name=subdir_name,
                subdir_path=root / subdir_name,
                plot_label=_plot_label_for_sample(sample_code, lepton),
                yield_category=_yield_category_for_sample(sample_code, lepton),
                is_primary_signal=(sample_code == CHANNELS[lepton]["primary_signal"]),
            )
        )
    return samples


def _required_fsr_read_columns(file_fields: set[str]) -> list[str]:
    required = [field for field in _fsr_keep_fields() if field in file_fields]
    for field in ("weight", "totalWeight"):
        if field in file_fields and field not in required:
            required.append(field)
    return required


def _count_subdir_events_for_fraction(subdir: Path) -> float:
    total = 0.0
    for parquet_file in sorted(subdir.rglob("*.parquet")):
        parquet_handle = pq.ParquetFile(parquet_file)
        file_fields = set(parquet_handle.schema_arrow.names)
        if "totalWeight" in file_fields:
            weights = ak.from_parquet(str(parquet_file), columns=["totalWeight"])["totalWeight"]
            total += float(ak.sum(weights))
        else:
            total += float(parquet_handle.metadata.num_rows)
    return total


def _truncate_chunk_to_budget(chunk: ak.Array, remaining_budget: float | None) -> tuple[ak.Array, float | None]:
    if remaining_budget is None or len(chunk) == 0:
        return chunk, remaining_budget

    if remaining_budget <= 0:
        return chunk[:0], 0.0

    field = weight_field(chunk)
    if field is None:
        if len(chunk) <= remaining_budget:
            return chunk, remaining_budget - float(len(chunk))
        cutoff = int(max(np.floor(remaining_budget), 0.0))
        return chunk[:cutoff], 0.0

    weights = np.asarray(ak.to_numpy(chunk[field]), dtype=float)
    used_budget = float(np.sum(weights))
    if used_budget <= remaining_budget:
        return chunk, remaining_budget - used_budget

    cumulative = np.cumsum(weights)
    cutoff = int(np.searchsorted(cumulative, remaining_budget, side="left")) + 1
    cutoff = min(cutoff, len(chunk))
    used_budget = float(np.sum(weights[:cutoff]))
    return chunk[:cutoff], max(remaining_budget - used_budget, 0.0)


def _ensure_charge_product(chunk: ak.Array) -> ak.Array:
    if "charge_product" in chunk.fields:
        return chunk
    return ak.with_field(chunk, chunk["lep_charge"][:, 0] * chunk["lep_charge"][:, 1], "charge_product")


def iterate_fsr_chunks(config: FSRStudyConfig, samples: list[FSRSample]):
    iterator = progress_iter(samples, total=len(samples), desc=f"{config.lepton} fsr study", unit="sample")
    for sample in iterator:
        remaining_budget: float | None = None
        if config.read_fraction < 1.0:
            total_budget = _count_subdir_events_for_fraction(sample.subdir_path)
            remaining_budget = total_budget * config.read_fraction

        for parquet_file in sorted(sample.subdir_path.rglob("*.parquet")):
            if remaining_budget is not None and remaining_budget <= 0:
                break

            parquet_handle = pq.ParquetFile(parquet_file)
            file_fields = set(parquet_handle.schema_arrow.names)
            missing = set(_fsr_keep_fields()) - file_fields
            if missing:
                raise KeyError(f"Missing required FSR fields in {parquet_file}: {sorted(missing)}")

            columns = _required_fsr_read_columns(file_fields)
            for row_group in range(parquet_handle.num_row_groups):
                if remaining_budget is not None and remaining_budget <= 0:
                    break

                table = parquet_handle.read_row_group(row_group, columns=columns)
                if table.num_rows == 0:
                    continue

                chunk = ak.from_arrow(table)
                chunk, remaining_budget = _truncate_chunk_to_budget(chunk, remaining_budget)
                if len(chunk) == 0:
                    continue

                chunk = _ensure_charge_product(chunk)
                chunk = chunk[chunk["charge_product"] < 0]
                if len(chunk) == 0:
                    continue

                yield sample, chunk


def _to_numpy(values: ak.Array) -> np.ndarray:
    return np.asarray(ak.to_numpy(values), dtype=float)


def _corrected_mass_from_dpt(
    pt: np.ndarray,
    eta: np.ndarray,
    phi: np.ndarray,
    energy: np.ndarray,
    dpt: np.ndarray,
) -> np.ndarray:
    pt_corr = pt + dpt
    energy_corr = energy + dpt * np.cosh(eta)

    px = pt_corr * np.cos(phi)
    py = pt_corr * np.sin(phi)
    pz = pt_corr * np.sinh(eta)

    total_energy = np.sum(energy_corr, axis=1)
    total_px = np.sum(px, axis=1)
    total_py = np.sum(py, axis=1)
    total_pz = np.sum(pz, axis=1)

    mass_squared = total_energy**2 - total_px**2 - total_py**2 - total_pz**2
    return np.sqrt(np.clip(mass_squared, 0.0, None))


def _leading_or_both_metric(values: np.ndarray, require_both: bool) -> np.ndarray:
    if require_both:
        return np.max(values, axis=1)
    return values[:, 0]


def add_fsr_proxy_fields_chunk(chunk: ak.Array, config: FSRStudyConfig) -> FSRChunkView:
    raw_mass = _to_numpy(chunk["mass"])
    pt = _to_numpy(chunk["lep_pt"])
    eta = _to_numpy(chunk["lep_eta"])
    phi = _to_numpy(chunk["lep_phi"])
    energy = _to_numpy(chunk["lep_e"])
    ptcone = _to_numpy(chunk["lep_ptvarcone30"])
    etcone = _to_numpy(chunk["lep_topoetcone20"])

    event_mask = raw_mass < config.fsr_apply_below_mass
    scaled_etcone = config.fsr_etcone_scale * etcone

    dpt_maxcone = np.zeros_like(etcone)
    if raw_mass.size:
        max_indices = np.argmax(etcone, axis=1)
        dpt_maxcone[np.arange(raw_mass.size), max_indices] = scaled_etcone[np.arange(raw_mass.size), max_indices]
    dpt_maxcone = np.where(event_mask[:, None], dpt_maxcone, 0.0)

    dpt_both = np.where(event_mask[:, None], scaled_etcone, 0.0)

    mass_fsr_maxcone = _corrected_mass_from_dpt(pt, eta, phi, energy, dpt_maxcone)
    mass_fsr_both = _corrected_mass_from_dpt(pt, eta, phi, energy, dpt_both)
    etcone_sub_maxcone = np.maximum(etcone - dpt_maxcone, 0.0)

    weights = np.ones(raw_mass.size, dtype=float)
    field = weight_field(chunk)
    if field is not None:
        weights = _to_numpy(chunk[field])

    return FSRChunkView(
        mass_raw=raw_mass,
        mass_fsr_maxcone=mass_fsr_maxcone,
        mass_fsr_both=mass_fsr_both,
        ptcone_metric=_leading_or_both_metric(ptcone, config.require_both_iso),
        etcone_metric_raw=_leading_or_both_metric(etcone, config.require_both_iso),
        etcone_metric_fsrsub_maxcone=_leading_or_both_metric(etcone_sub_maxcone, config.require_both_iso),
        weights=weights,
    )


def build_chunk_selection_masks(chunk_view: FSRChunkView, config: FSRStudyConfig) -> dict[str, np.ndarray]:
    pass_nominal = (
        (chunk_view.ptcone_metric < config.nominal_iso.ptcone_max)
        & (chunk_view.etcone_metric_raw < config.nominal_iso.etcone_max)
    )
    pass_loose = (
        (chunk_view.ptcone_metric < config.loose_iso.ptcone_max)
        & (chunk_view.etcone_metric_raw < config.loose_iso.etcone_max)
    )
    return {
        "pass_nominal": pass_nominal,
        "pass_loose": pass_loose,
        "between": (~pass_nominal) & pass_loose,
    }


def build_plot_specs(config: FSRStudyConfig) -> dict[str, PlotSpec]:
    nominal = config.nominal_iso
    loose = config.loose_iso
    return {
        "between_raw_full": PlotSpec(
            selection_name="between",
            mass_field="mass_raw",
            plot_config=config.mass_plots["FULL"],
            title=(
                f"OS dimuon: fail nominal ({nominal.ptcone_max:.2f},{nominal.etcone_max:.2f}) "
                f"but pass loose ({loose.ptcone_max:.2f},{loose.etcone_max:.2f}) [raw mass]"
            ),
        ),
        "between_fsr_maxcone_full": PlotSpec(
            selection_name="between",
            mass_field="mass_fsr_maxcone",
            plot_config=config.mass_plots["FULL"],
            title="Same control sample with toy FSR recovery [maxcone mode]",
        ),
        "between_fsr_both_full": PlotSpec(
            selection_name="between",
            mass_field="mass_fsr_both",
            plot_config=config.mass_plots["FULL"],
            title="Same control sample with toy FSR recovery [both-muons mode]",
        ),
        "between_raw_zoom": PlotSpec(
            selection_name="between",
            mass_field="mass_raw",
            plot_config=config.mass_plots["ZOOM"],
            title="Control sample near Z peak [raw mass]",
        ),
        "between_fsr_maxcone_zoom": PlotSpec(
            selection_name="between",
            mass_field="mass_fsr_maxcone",
            plot_config=config.mass_plots["ZOOM"],
            title="Control sample near Z peak [FSR recovery: maxcone]",
        ),
        "between_fsr_both_zoom": PlotSpec(
            selection_name="between",
            mass_field="mass_fsr_both",
            plot_config=config.mass_plots["ZOOM"],
            title="Control sample near Z peak [FSR recovery: both]",
        ),
        "pass_loose_raw_zoom": PlotSpec(
            selection_name="pass_loose",
            mass_field="mass_raw",
            plot_config=config.mass_plots["ZOOM"],
            title=f"Pass loose isolation ({loose.ptcone_max:.2f},{loose.etcone_max:.2f}) [raw mass]",
        ),
        "pass_loose_fsr_maxcone_zoom": PlotSpec(
            selection_name="pass_loose",
            mass_field="mass_fsr_maxcone",
            plot_config=config.mass_plots["ZOOM"],
            title="Pass loose isolation with toy FSR recovery [maxcone]",
        ),
        "pass_loose_fsr_both_zoom": PlotSpec(
            selection_name="pass_loose",
            mass_field="mass_fsr_both",
            plot_config=config.mass_plots["ZOOM"],
            title="Pass loose isolation with toy FSR recovery [both]",
        ),
    }


def initialise_plot_accumulators(plot_specs: dict[str, PlotSpec], plot_labels: list[str]) -> dict[str, HistogramAccumulator]:
    return {
        plot_name: HistogramAccumulator.from_plot_config(spec.plot_config, plot_labels)
        for plot_name, spec in plot_specs.items()
    }


def initialise_window_totals(config: FSRStudyConfig) -> dict[str, list[YieldSummary]]:
    return {mass_field: [YieldSummary() for _ in config.windows] for mass_field in WINDOW_MASS_FIELDS}


def build_scan_grid(config: FSRStudyConfig) -> ScanGrid:
    if not config.scan.run:
        return ScanGrid(points=(), pt_thresholds=(), et_thresholds=())

    points = tuple(
        generate_scan_points(
            nominal_ptcone=config.nominal_iso.ptcone_max,
            nominal_etcone=config.nominal_iso.etcone_max,
            scan_mode=config.scan.mode,
            ptcone_range=config.scan.ptcone_range,
            ptcone_step=config.scan.ptcone_step,
            etcone_range=config.scan.etcone_range,
            etcone_step=config.scan.etcone_step,
            local_box_ptcone_half_width=config.scan.local_box_ptcone_half_width,
            local_box_etcone_half_width=config.scan.local_box_etcone_half_width,
        )
    )
    return ScanGrid(
        points=points,
        pt_thresholds=tuple(sorted({float(point[0]) for point in points})),
        et_thresholds=tuple(sorted({float(point[1]) for point in points})),
    )


def initialise_sigma_accumulators(scan_grid: ScanGrid) -> dict[str, SigmaStudyAccumulator]:
    return {
        study_name: SigmaStudyAccumulator.create(spec, len(scan_grid.points))
        for study_name, spec in SCAN_STUDIES.items()
    }


def _threshold_mask_cache(values: np.ndarray, thresholds: tuple[float, ...]) -> dict[float, np.ndarray]:
    return {float(threshold): values < float(threshold) for threshold in thresholds}


def accumulate_between_nominal_loose_histograms(
    plot_specs: dict[str, PlotSpec],
    plot_accumulators: dict[str, HistogramAccumulator],
    sample: FSRSample,
    chunk_view: FSRChunkView,
    selection_masks: dict[str, np.ndarray],
) -> None:
    for plot_name, spec in plot_specs.items():
        mask = selection_masks[spec.selection_name]
        if not np.any(mask):
            continue

        values = getattr(chunk_view, spec.mass_field)[mask]
        weights = chunk_view.weights[mask]
        plot_accumulators[plot_name].fill(sample.plot_label, values, weights)


def accumulate_between_mass_windows(
    window_totals: dict[str, list[YieldSummary]],
    sample: FSRSample,
    chunk_view: FSRChunkView,
    between_mask: np.ndarray,
    config: FSRStudyConfig,
) -> None:
    if not np.any(between_mask):
        return

    selected_weights = chunk_view.weights[between_mask]
    for mass_field, summaries in window_totals.items():
        selected_mass = getattr(chunk_view, mass_field)[between_mask]
        for index, (low_edge, high_edge) in enumerate(config.windows):
            window_mask = (selected_mass > low_edge) & (selected_mass < high_edge)
            summaries[index].add(sample, window_mask, selected_weights)


def accumulate_sigma_scan(
    sigma_accumulators: dict[str, SigmaStudyAccumulator],
    scan_grid: ScanGrid,
    sample: FSRSample,
    chunk_view: FSRChunkView,
    config: FSRStudyConfig,
) -> None:
    if not scan_grid.points:
        return

    pt_masks = _threshold_mask_cache(chunk_view.ptcone_metric, scan_grid.pt_thresholds)
    raw_et_masks = _threshold_mask_cache(chunk_view.etcone_metric_raw, scan_grid.et_thresholds)
    fsrsub_et_masks = _threshold_mask_cache(chunk_view.etcone_metric_fsrsub_maxcone, scan_grid.et_thresholds)

    mass_low, mass_high = config.mass_window
    mass_masks = {
        study_name: (
            (getattr(chunk_view, study.spec.mass_field) > mass_low)
            & (getattr(chunk_view, study.spec.mass_field) < mass_high)
        )
        for study_name, study in sigma_accumulators.items()
    }

    for study_name, accumulator in sigma_accumulators.items():
        et_masks = raw_et_masks
        if accumulator.spec.etcone_metric_key == "etcone_metric_fsrsub_maxcone":
            et_masks = fsrsub_et_masks

        base_mass_mask = mass_masks[study_name]
        for index, (ptcone_max, etcone_max) in enumerate(scan_grid.points):
            selected_mask = base_mass_mask & pt_masks[float(ptcone_max)] & et_masks[float(etcone_max)]
            if not np.any(selected_mask):
                continue

            if sample.yield_category == "data":
                accumulator.data_counts[index] += float(np.count_nonzero(selected_mask))
                continue

            selected_weights = chunk_view.weights[selected_mask]
            if sample.is_primary_signal:
                accumulator.signal_weights[index] += float(np.sum(selected_weights))
                continue

            accumulator.background_weights[index] += float(np.sum(selected_weights))
            accumulator.background_vars[index] += float(np.sum(np.square(selected_weights)))


def build_mass_window_table(window_totals: dict[str, list[YieldSummary]], config: FSRStudyConfig) -> pd.DataFrame:
    rows: list[dict] = []
    for mass_field in WINDOW_MASS_FIELDS:
        for (low_edge, high_edge), summary in zip(config.windows, window_totals[mass_field]):
            rows.append(
                {
                    "mass_field": mass_field,
                    "window": f"{low_edge:.0f}-{high_edge:.0f}",
                    **summary.as_dict(),
                }
            )
    return pd.DataFrame(rows)


def finalize_sigma_scan(
    *,
    sigma_accumulators: dict[str, SigmaStudyAccumulator],
    scan_grid: ScanGrid,
    config: FSRStudyConfig,
    backend: dict,
) -> pd.DataFrame:
    if not scan_grid.points:
        return pd.DataFrame()

    primary_signal = CHANNELS[config.lepton]["primary_signal"]
    produced_sumw_cache: dict[str, float] = {}
    signal_total_weight = produced_sumw(
        backend["produced_event_count"],
        primary_signal,
        LUMI_FB,
        cache=produced_sumw_cache,
    )

    rows: list[dict] = []
    iterator = progress_iter(scan_grid.points, total=len(scan_grid.points), desc="fsr sigma finalize", unit="pt")
    for index, (ptcone_max, etcone_max) in enumerate(iterator):
        row = {
            "ptcone_max": float(ptcone_max),
            "etcone_max": float(etcone_max),
            "is_nominal": bool(
                np.isclose(ptcone_max, config.nominal_iso.ptcone_max)
                and np.isclose(etcone_max, config.nominal_iso.etcone_max)
            ),
        }
        for study_name, accumulator in sigma_accumulators.items():
            try:
                result = compute_sigma_from_totals(
                    primary_signal=primary_signal,
                    n_selected=accumulator.data_counts[index],
                    n_background_mc=accumulator.background_weights[index],
                    n_background_var=accumulator.background_vars[index],
                    signal_pass_weight=accumulator.signal_weights[index],
                    signal_total_weight=signal_total_weight,
                    extra_bkg=0.0,
                )
                row[f"sigma_pb_{study_name}"] = float(result["sigma_pb"])
                row[f"epsilon_{study_name}"] = float(result["epsilon"])
                row[f"n_sig_data_{study_name}"] = float(result["N_sig_data"])
                row[f"n_selected_{study_name}"] = float(result["N_selected"])
                row[f"n_bkg_mc_{study_name}"] = float(result["N_bkg_mc"])
                row[f"dsigma_stat_pb_{study_name}"] = float(result["dsigma_stat_pb"])
                row[f"dsigma_lumi_pb_{study_name}"] = float(result["dsigma_lumi_pb"])
            except Exception as exc:
                row[f"sigma_pb_{study_name}"] = np.nan
                row[f"epsilon_{study_name}"] = np.nan
                row[f"n_sig_data_{study_name}"] = np.nan
                row[f"n_selected_{study_name}"] = np.nan
                row[f"n_bkg_mc_{study_name}"] = np.nan
                row[f"dsigma_stat_pb_{study_name}"] = np.nan
                row[f"dsigma_lumi_pb_{study_name}"] = np.nan
                row[f"error_{study_name}"] = str(exc)
        rows.append(row)

    scan_df = pd.DataFrame(rows)
    for study_name in sigma_accumulators:
        sigma_col = f"sigma_pb_{study_name}"
        if sigma_col not in scan_df.columns:
            continue
        nominal_rows = scan_df[scan_df["is_nominal"] & np.isfinite(scan_df[sigma_col])]
        if nominal_rows.empty:
            scan_df[f"sigma_abs_shift_pb_{study_name}"] = np.nan
            scan_df[f"sigma_frac_shift_{study_name}"] = np.nan
            continue
        nominal_sigma = float(nominal_rows.iloc[0][sigma_col])
        abs_shift = (scan_df[sigma_col] - nominal_sigma).abs()
        frac_shift = abs_shift / abs(nominal_sigma) if nominal_sigma != 0 else np.nan
        scan_df[f"sigma_abs_shift_pb_{study_name}"] = abs_shift
        scan_df[f"sigma_frac_shift_{study_name}"] = frac_shift
    return scan_df


def save_scan_outputs(scan_df: pd.DataFrame, output_dir: Path, config: FSRStudyConfig) -> None:
    if scan_df.empty:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    scan_df.to_csv(output_dir / "sigma_scan.csv", index=False)
    write_json(output_dir / "sigma_scan.json", scan_df.to_dict(orient="records"))

    for study_name, spec in SCAN_STUDIES.items():
        sigma_col = f"sigma_pb_{study_name}"
        if sigma_col not in scan_df.columns:
            continue
        save_scan_heatmap(
            scan_df,
            output_dir / f"heatmap_{study_name}.png",
            value_column=sigma_col,
            value_label="sigma_pb [pb]",
            title=spec.title,
            nominal_ptcone=config.nominal_iso.ptcone_max,
            nominal_etcone=config.nominal_iso.etcone_max,
        )
        shift_col = f"sigma_frac_shift_{study_name}"
        if shift_col in scan_df.columns:
            save_scan_heatmap(
                scan_df,
                output_dir / f"heatmap_fracshift_{study_name}.png",
                value_column=shift_col,
                value_label="fractional |Δsigma|",
                title=f"fractional |Δsigma| ({study_name})",
                nominal_ptcone=config.nominal_iso.ptcone_max,
                nominal_etcone=config.nominal_iso.etcone_max,
            )


def save_stacked_mass_plot(
    *,
    accumulator: HistogramAccumulator,
    plot_labels: list[str],
    output_path: Path,
    title: str,
    logy: bool,
) -> None:
    centres = 0.5 * (accumulator.edges[:-1] + accumulator.edges[1:])
    widths = np.diff(accumulator.edges)

    figure, (main_axis, ratio_axis) = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    palette = {
        "Data": "black",
        plot_labels[1]: "tab:blue",
    }
    fallback_colors = ["tab:olive", "tab:green", "tab:red", "tab:purple", "tab:cyan", "tab:orange"]

    mc_labels = [label for label in plot_labels if label != "Data"]
    for index, label in enumerate(mc_labels[1:], start=0):
        palette.setdefault(label, fallback_colors[index % len(fallback_colors)])

    stacked_total = np.zeros_like(centres)
    stacked_var = np.zeros_like(centres)
    for label in mc_labels:
        counts = accumulator.counts_by_label[label]
        main_axis.bar(
            centres,
            counts,
            width=widths,
            bottom=stacked_total,
            align="center",
            color=palette.get(label, "tab:gray"),
            alpha=0.8,
            edgecolor="black",
            linewidth=0.3,
            label=label,
        )
        stacked_total += counts
        stacked_var += accumulator.variances_by_label[label]

    mc_unc = np.sqrt(stacked_var)
    if np.any(mc_unc > 0):
        main_axis.stairs(
            stacked_total + mc_unc,
            accumulator.edges,
            baseline=np.clip(stacked_total - mc_unc, 0.0, None),
            fill=True,
            color="gray",
            alpha=0.18,
            label="MC stat.",
        )

    data_counts = accumulator.counts_by_label["Data"]
    data_err = np.sqrt(accumulator.variances_by_label["Data"])
    main_axis.errorbar(
        centres,
        data_counts,
        yerr=data_err,
        fmt="ko",
        markersize=4,
        linewidth=1.0,
        label="Data",
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(data_counts, stacked_total, out=np.full_like(data_counts, np.nan), where=stacked_total > 0)
        ratio_err = np.divide(data_err, stacked_total, out=np.full_like(data_err, np.nan), where=stacked_total > 0)
        mc_frac_unc = np.divide(mc_unc, stacked_total, out=np.zeros_like(mc_unc), where=stacked_total > 0)

    ratio_axis.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    ratio_axis.errorbar(centres, ratio, yerr=ratio_err, fmt="ko", markersize=4, linewidth=1.0)
    if np.any(mc_frac_unc > 0):
        ratio_axis.stairs(
            1.0 + mc_frac_unc,
            accumulator.edges,
            baseline=np.clip(1.0 - mc_frac_unc, 0.0, None),
            fill=True,
            color="gray",
            alpha=0.18,
        )

    main_axis.set_ylabel("Events")
    ratio_axis.set_ylabel("Data/MC")
    ratio_axis.set_xlabel("mass [GeV]")
    main_axis.set_title(title)

    if logy:
        main_axis.set_yscale("log")
        positive_values = np.concatenate(
            [
                data_counts[data_counts > 0],
                stacked_total[stacked_total > 0],
            ]
        )
        ymin = 0.1
        ymax = 10.0
        if positive_values.size:
            ymax = max(positive_values.max() * 10.0, 1.0)
            ymin = max(min(positive_values.min() / 3.0, 0.1), 1e-3)
        main_axis.set_ylim(ymin, ymax)

    finite_ratio = ratio[np.isfinite(ratio)]
    if finite_ratio.size:
        finite_mask = np.isfinite(ratio) & np.isfinite(ratio_err)
        upper = ratio[finite_mask] + ratio_err[finite_mask]
        ratio_axis.set_ylim(0.0, max(2.0, float(np.nanmax(upper)) * 1.2 if upper.size else 2.0))
    else:
        ratio_axis.set_ylim(0.0, 2.0)

    main_axis.grid(alpha=0.2)
    ratio_axis.grid(alpha=0.2)

    handles, labels = main_axis.get_legend_handles_labels()
    if "Data" in labels:
        data_index = labels.index("Data")
        order = [data_index] + [index for index in range(len(labels)) if index != data_index]
        handles = [handles[index] for index in order]
        labels = [labels[index] for index in order]
    main_axis.legend(handles, labels, frameon=False, fontsize=9)

    figure.tight_layout()
    save_fig(figure, output_path)


def render_all_mass_plots(
    plot_specs: dict[str, PlotSpec],
    plot_accumulators: dict[str, HistogramAccumulator],
    plot_labels: list[str],
    plots_dir: Path,
    config: FSRStudyConfig,
) -> None:
    if not config.save_plots:
        return

    plots_dir.mkdir(parents=True, exist_ok=True)
    for plot_name, spec in plot_specs.items():
        save_stacked_mass_plot(
            accumulator=plot_accumulators[plot_name],
            plot_labels=plot_labels,
            output_path=plots_dir / f"{plot_name}.png",
            title=spec.title,
            logy=spec.plot_config.logy,
        )


def nominal_sigma_summary_lines(scan_df: pd.DataFrame) -> list[str]:
    if scan_df.empty:
        return ["Sigma scan was disabled or produced no rows."]

    nominal_rows = scan_df[scan_df["is_nominal"]]
    if nominal_rows.empty:
        return ["Sigma scan did not contain the nominal point."]

    nominal_row = nominal_rows.iloc[0]
    lines = ["Nominal sigma values:"]
    for study_name in SCAN_STUDIES:
        sigma_key = f"sigma_pb_{study_name}"
        if sigma_key in nominal_row and np.isfinite(nominal_row[sigma_key]):
            lines.append(f"- {study_name}: {float(nominal_row[sigma_key]):.6g} pb")
    return lines


def run() -> Path:
    ensure_script_directory()
    ensure_environment()
    backend = import_backend()
    config = resolve_fsr_config()

    run_root = (Path(__file__).resolve().parent / config.output_dir / f"run_{now_stamp()}").resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(run_root / "fsr_settings.json", asdict(config))

    root = ensure_fsr_tight_parquet(config, backend)
    manifest = read_manifest(root)
    if manifest is None:
        raise RuntimeError(f"Missing manifest under {root}")

    samples = resolve_fsr_samples(root, manifest, config.lepton)
    if not samples:
        raise RuntimeError(f"No FSR samples found under {root}")

    plot_labels = ordered_plot_labels(config.lepton)
    plot_specs = build_plot_specs(config)
    plot_accumulators = initialise_plot_accumulators(plot_specs, plot_labels)
    between_summary = YieldSummary()
    window_totals = initialise_window_totals(config)
    scan_grid = build_scan_grid(config)
    sigma_accumulators = initialise_sigma_accumulators(scan_grid)

    log_step(f"[{config.lepton}] Streaming FSR study chunks")
    for sample, chunk in iterate_fsr_chunks(config, samples):
        chunk_view = add_fsr_proxy_fields_chunk(chunk, config)
        selection_masks = build_chunk_selection_masks(chunk_view, config)

        between_mask = selection_masks["between"]
        between_summary.add(sample, between_mask, chunk_view.weights)
        accumulate_between_nominal_loose_histograms(
            plot_specs,
            plot_accumulators,
            sample,
            chunk_view,
            selection_masks,
        )
        accumulate_between_mass_windows(window_totals, sample, chunk_view, between_mask, config)
        accumulate_sigma_scan(sigma_accumulators, scan_grid, sample, chunk_view, config)

    plots_dir = run_root / "plots"
    render_all_mass_plots(plot_specs, plot_accumulators, plot_labels, plots_dir, config)

    window_table = build_mass_window_table(window_totals, config)
    window_table.to_csv(run_root / "between_mass_windows.csv", index=False)

    scan_df = finalize_sigma_scan(
        sigma_accumulators=sigma_accumulators,
        scan_grid=scan_grid,
        config=config,
        backend=backend,
    )
    scan_dir = run_root / "sigma_scan"
    save_scan_outputs(scan_df, scan_dir, config)

    summary_lines = [
        f"FSR toy study run directory: {run_root}",
        f"FSR tight parquet root: {root}",
        "",
        "Processing model:",
        "- streamed sample-by-sample and row-group-by-row-group from the dedicated FSR parquet",
        "- no full all-sample OS event dictionary is materialised in memory",
        "- stacked plots are built from accumulated histogram bins, not full event arrays",
        "",
        "Physics reminder:",
        "- mass_fsr_maxcone adds topoetcone20 back only to the muon with larger topoetcone20,",
        "  and only when the raw dimuon mass is below APPLY_BELOW_MASS.",
        "- masscorr_etsub_maxcone also subtracts that recovered proxy from the etcone",
        "  used in the selection, so it is the direct toy test of an FSR-aware isolation idea.",
        "",
        "Nominal / loose isolation:",
        f"- nominal: ptcone < {config.nominal_iso.ptcone_max:.3f}, etcone < {config.nominal_iso.etcone_max:.3f}",
        f"- loose:   ptcone < {config.loose_iso.ptcone_max:.3f}, etcone < {config.loose_iso.etcone_max:.3f}",
        "",
        "Control sample summary (fail nominal, pass loose):",
        f"{between_summary.as_dict()}",
        "",
        *nominal_sigma_summary_lines(scan_df),
        "",
        "Key files to inspect first:",
        f"- {plots_dir / 'between_raw_zoom.png'}",
        f"- {plots_dir / 'between_fsr_maxcone_zoom.png'}",
        f"- {plots_dir / 'between_fsr_both_zoom.png'}",
        f"- {scan_dir / 'heatmap_raw.png'}",
        f"- {scan_dir / 'heatmap_masscorr_maxcone.png'}",
        f"- {scan_dir / 'heatmap_masscorr_etsub_maxcone.png'}",
    ]
    write_text(run_root / "summary.txt", "\n".join(summary_lines))

    return run_root


if __name__ == "__main__":
    output = run()
    print(f"FSR study outputs saved under: {output}")
