from __future__ import annotations

from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


SETTINGS = {
    # Usually leave this alone unless you want both channels in one run.
    "LEPTONS": ("mu",),

    # Fraction read from the already-built tight parquet.
    "FRACTION": 1.0,

    # Baseline preselection before the tight parquet is written.
    "PT_MIN": 10.0,

    # Final signal-region mass selection for the nominal cut point.
    "MASS_WINDOW": (66.0, 116.0),
    "REQUIRE_BOTH_ISO": True,
    "FIXED_ISO": {"ptcone_max": 4.5, "etcone_max": 9.25},

    # Isolation scan:
    # - optimisation uses significance with an OS signal-efficiency floor,
    # - diagnostics enrich the scan with DD-background-aware sigma values.
    "ISO_SCAN": {
        "RUN_OPTIMISATION_SCAN": True,
        "RUN_SCAN_DIAGNOSTICS": True,
        "PTCONE_RANGE": (0.0, 10.0),
        "PTCONE_STEP": 1.0,
        "ETCONE_RANGE": (0.0, 20.0),
        "ETCONE_STEP": 1.0,
        # Usually leave this alone; it prevents the scan from choosing
        # isolation cuts that destroy the OS signal efficiency.
        "OS_SIG_EFF_MIN": {"mu": 0.995, "e": 0.995},
        # These save flags only do anything if the corresponding scan is enabled.
        "SAVE_HEATMAPS": True,
        "SAVE_3D_PLOTS": True,
        "SAVE_SLICE_PLOTS": True,
        "MONOTONICITY_TOL_ABS": 1e-4,
    },

    # Keep the terminal concise by default and show progress bars for the
    # genuinely slow stages.
    "TERMINAL": {
        "QUIET_BACKEND_OUTPUT": True,
        "SHOW_STAGE_LOGS": True,
    },
    "PROGRESS": {
        "ENABLED": True,
    },

    # If APPLY is True, require the Medium-ID field whenever it exists.
    # This applies to both data and MC; no extra scope setting is exposed.
    "MEDIUM_ID": {
        "APPLY": True,
        "FIELD_CANDIDATES": ["lep_isMediumID"],
    },

    # Keep the notebook-style memory-friendly tight-parquet design.
    "USE_TIGHT_PARQUET": True,
    "TIGHT_PARQUET": {
        "BUILD_FRACTION": 1.0,
        "FORCE_REBUILD": False,
        "ROOT_DIR": "../../tight-parquet",
        "MAIN_TAG": None,
        "CONTROL_TAG": None,
        "MAIN_STAGE": "baseline",
        "MAIN_KEEP_FIELDS": [
            "lep_pt",
            "lep_ptvarcone30",
            "lep_topoetcone20",
            "mass",
            "charge_product",
        ],
        "CONTROL_KEEP_FIELDS": [
            "lep_pt",
            "lep_type",
            "lep_charge",
            "lep_ptvarcone30",
            "lep_topoetcone20",
            "mass",
            "charge_product",
        ],
        "WRITE_METADATA": True,
    },

    # Usually the defaults here are fine. The code always reports
    # none, wrong_flavour, wrong_charge, and both_average together.
    "ADDITIONAL_DATA_DRIVEN_BKG": {
        "ENABLED": True,
        "CLIP_NEGATIVE_TO_ZERO": True,
        "USE_CONTROL_TIGHT_PARQUET": True,
        "FORCE_REBUILD": False,
        "CONTROL_TRIGGER_MODE": "or",
        "SAVE_DEBUG_TABLES": True,
        "SAVE_PLOTS": True,
        "SAVE_JSON": True,
    },

    # Keep mass-window variations safely above the 40 GeV threshold
    # of the primary Z signal samples.
    "SYSTEMATICS": {
        "MASS_WINDOW_VARIATIONS": [
            (68.0, 114.0),
            (64.0, 118.0),
            (70.0, 112.0),
            (62.0, 120.0),
        ],
    },

    "PLOTS": {
        "LEADING_PT": {"xmin": 0, "xmax": 200, "bins": 50, "logy": True},
        "MASS_FULL": {"xmin": 0, "xmax": 200, "bins": 120, "logy": True},
        "MASS_ZOOM": {"xmin": 60, "xmax": 120, "bins": 60, "logy": True},
    },

    "OUTPUT_DIR": "output_py",
    "SAVE_PLOTS": True,
    "SAVE_TABLES": True,
    "SAVE_JSON": True,
    "MAKE_GROUP_FIGURES": True,
    "AUTO_INSTALL": False,
    "RUN_INSTALL_FROM_ENV_YML": False,
}


CHANNELS = {
    "mu": {
        "string_codes": ["2to4lep", "Zmumu", "m10_40_Zmumu", "Ztautau", "ttbar", "Wmunu"],
        "type_sum": 26,
        "trigger_field": "trigM",
        "primary_signal": "Zmumu",
        "signal_samples": ["Zmumu", "m10_40_Zmumu"],
        "background_samples": ["Ztautau", "ttbar", "Wmunu"],
        "leading_label": r"leading muon $p_T$ [GeV]",
        "title_token": "mu",
    },
    "e": {
        "string_codes": ["2to4lep", "Zee", "m10_40_Zee", "Ztautau", "ttbar", "Wenu"],
        "type_sum": 22,
        "trigger_field": "trigE",
        "primary_signal": "Zee",
        "signal_samples": ["Zee", "m10_40_Zee"],
        "background_samples": ["Ztautau", "ttbar", "Wenu"],
        "leading_label": r"leading electron $p_T$ [GeV]",
        "title_token": "e",
    },
}


RAW_BUILD_VARS_COMMON = [
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


LUMI_FB = 30.6
LUMI_REL_UNC = 0.017
LUMI_PB = LUMI_FB * 1000.0


DD_ESTIMATOR_METHODS = ("wrong_flavour", "wrong_charge", "both_average")
DD_METHODS = ("none", "wrong_flavour", "wrong_charge", "both_average")


def fraction_label(frac: float | int) -> str:
    return str(float(frac)).replace(".", "_")


def label_float(value: float) -> str:
    return f"{value:.1f}".replace(".", "p")


def medium_id_mode() -> str:
    return "apply_all_if_available" if SETTINGS["MEDIUM_ID"]["APPLY"] else "disabled"


def iso_eff_threshold_for(lepton: str) -> float:
    value = SETTINGS["ISO_SCAN"]["OS_SIG_EFF_MIN"]
    if isinstance(value, dict):
        return float(value[lepton])
    return float(value)


def control_sample_codes_for_build() -> list[str]:
    ordered: list[str] = []
    for lepton in SETTINGS["LEPTONS"]:
        for sample_code in CHANNELS[lepton]["string_codes"]:
            if sample_code not in ordered:
                ordered.append(sample_code)
    if "2to4lep" not in ordered:
        ordered.insert(0, "2to4lep")
    return ordered or ["2to4lep"]


def user_facing_sample_label(sample_code: str) -> str:
    if sample_code in {"m10_40_Zee", "m10_40_Zmumu"}:
        return f"Signal (LowMassDY {sample_code})"
    return sample_code
