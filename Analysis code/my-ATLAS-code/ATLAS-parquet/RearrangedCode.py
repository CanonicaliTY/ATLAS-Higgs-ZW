# =========================
# CLEAN DILEPTON ANALYSIS (v2)
# Change LEPTON = "mu" -> "e" to switch channel
# =========================

import numpy as np
import pandas as pd
import awkward as ak
import vector
import sys

# If backend import is not already in scope, keep this:
if "../" not in sys.path:
    sys.path.append("../")

from backend import validate_read_variables, analysis_parquet, plot_stacked_hist

# -------------------------
# 0) User config
# -------------------------
LEPTON = "mu"            # "mu" or "e"
FRACTION = 0.02          # load fraction
PT_MIN = 10.0            # baseline lepton pT cut (GeV)
REQUIRE_BOTH_ISO = True  # iso cut on both leptons

# Mass window you want to study (typical Z window)
MASS_WINDOW = (66.0, 116.0)

# For iso scan (float step + range)
PTCONE_RANGE = (0.0, 10.0)   # GeV
PTCONE_STEP  = 0.25          # GeV
ETCONE_RANGE = (0.0, 10.0)   # GeV
ETCONE_STEP  = 0.25          # GeV

# Minimum OS signal efficiency requirement inside MASS_WINDOW
OS_SIG_EFF_MIN = 0.995

# -------------------------
# 1) Channel configuration
# -------------------------
CHANNELS = {
    "mu": {
        "string_codes": ["2to4lep", "Zmumu", "m10_40_Zmumu", "Ztautau", "ttbar", "Wmunu"],
        "type_sum": 26,   # 13+13
        "trigger_field": "trigM",
        "signals": ["Zmumu", "m10_40_Zmumu"],
        "bkgs":    ["Ztautau", "ttbar", "Wmunu"],
        "leading_label": "leading muon pT [GeV]",
    },
    "e": {
        "string_codes": ["2to4lep", "Zee", "m10_40_Zee", "Ztautau", "ttbar", "Wenu"],
        "type_sum": 22,   # 11+11
        "trigger_field": "trigE",
        "signals": ["Zee", "m10_40_Zee"],
        "bkgs":    ["Ztautau", "ttbar", "Wenu"],
        "leading_label": "leading electron pT [GeV]",
    }
}

cfg = CHANNELS[LEPTON]

# -------------------------
# 2) Base selection + OS/SS split (self-contained)
# -------------------------
def base_dilepton(data, lepton="mu", pt_min=10.0):
    """Baseline: exactly 2 leptons, correct flavor, pT thresholds, trigger, compute mass."""
    cfg_local = CHANNELS[lepton]

    data = data[data["lep_n"] == 2]
    data = data[(data["lep_type"][:,0] + data["lep_type"][:,1]) == cfg_local["type_sum"]]
    data = data[(data["lep_pt"][:,0] > pt_min) & (data["lep_pt"][:,1] > pt_min)]
    data = data[data[cfg_local["trigger_field"]]]

    p4 = vector.zip({
        "pt":  data["lep_pt"],
        "eta": data["lep_eta"],
        "phi": data["lep_phi"],
        "E":   data["lep_e"],
    })
    data["mass"] = (p4[:,0] + p4[:,1]).M
    return data

def make_cut_function(lepton="mu", sign="OS", pt_min=10.0):
    """Return a cut_function(data)->data for analysis_parquet."""
    def _cut(data):
        data = base_dilepton(data, lepton=lepton, pt_min=pt_min)
        qprod = data["lep_charge"][:,0] * data["lep_charge"][:,1]
        if sign.upper() == "OS":
            return data[qprod == -1]
        elif sign.upper() == "SS":
            return data[qprod == +1]
        else:
            raise ValueError("sign must be 'OS' or 'SS'")
    return _cut


# -------------------------
# 3) Variables to read (must include iso variables!)
# -------------------------
base_vars = [
    "lep_n", "lep_pt", "lep_eta", "lep_phi", "lep_e",
    "lep_type", "lep_charge",
    "trigM", "trigE",
]

iso_vars = [
    "lep_ptvarcone30",     # track isolation
    "lep_topoetcone20",    # calo isolation
]

read_vars = validate_read_variables(cfg["string_codes"], base_vars + iso_vars)

# -------------------------
# 4) Load OS & SS
# -------------------------
cut_OS = make_cut_function(lepton=LEPTON, sign="OS", pt_min=PT_MIN)
cut_SS = make_cut_function(lepton=LEPTON, sign="SS", pt_min=PT_MIN)

data_OS = analysis_parquet(read_vars, cfg["string_codes"], fraction=FRACTION, cut_function=cut_OS)
data_SS = analysis_parquet(read_vars, cfg["string_codes"], fraction=FRACTION, cut_function=cut_SS)

# -------------------------
# 5) Robustly infer the fraction key suffix (avoid formatting mismatch)
# -------------------------
def infer_suffix(d, data_code="2to4lep"):
    for k in d.keys():
        if k.startswith(data_code + "_"):
            return k.split(data_code + "_", 1)[1]
    raise KeyError("Cannot infer suffix: no 2to4lep_* key found.")

suffix = infer_suffix(data_OS, "2to4lep")
print("Inferred key suffix =", suffix)

def get_sample(d, code):
    return d.get(f"{code}_{suffix}")

# -------------------------
# 6) Build plot_dict in a fixed order (important for color_list)
# -------------------------
def build_plot_dict(d, lepton="mu"):
    cfg_local = CHANNELS[lepton]
    plot_dict = {
        "Data": get_sample(d, "2to4lep"),
    }
    for s in cfg_local["signals"]:
        plot_dict[f"Signal {s}"] = get_sample(d, s)
    for b in cfg_local["bkgs"]:
        plot_dict[f"Background {b}"] = get_sample(d, b)
    return plot_dict

plot_OS = build_plot_dict(data_OS, LEPTON)
plot_SS = build_plot_dict(data_SS, LEPTON)

# plot colors: Data (k), Signal1 (b), Signal2 (y), Backgrounds (g,r,m)
color_list = ["k", "b", "y", "g", "r", "m"]


# -------------------------
# 7) BEFORE plots (OS & SS): leading pT and mass
# -------------------------

# pT (leading lepton)
plot_variable = "lep_pt[0]"
xmin, xmax = 0, 200
num_bins = 50

fig_os_pt, _ = plot_stacked_hist(
    plot_OS, plot_variable, color_list,
    num_bins, xmin, xmax, cfg["leading_label"] + f" ({LEPTON} OS, before cuts)",
    logy=True, show_text=True, residual_plot=True, save_fig=False
)

fig_ss_pt, _ = plot_stacked_hist(
    plot_SS, plot_variable, color_list,
    num_bins, xmin, xmax, cfg["leading_label"] + f" ({LEPTON} SS, before cuts)",
    logy=True, show_text=True, residual_plot=True, save_fig=False
)

# mass (full range)
plot_variable = "mass"
xmin, xmax = 0, 200
num_bins = 120

fig_os_mass_before, _ = plot_stacked_hist(
    plot_OS, plot_variable, color_list,
    num_bins, xmin, xmax, f"mass [GeV] ({LEPTON} OS, before cuts)",
    logy=True, show_text=True, residual_plot=True, save_fig=False
)

fig_ss_mass_before, _ = plot_stacked_hist(
    plot_SS, plot_variable, color_list,
    num_bins, xmin, xmax, f"mass [GeV] ({LEPTON} SS, before cuts)",
    logy=True, show_text=True, residual_plot=True, save_fig=False
)


# -------------------------
# 8) Selection utilities: mass window + ptcone + etcone
# -------------------------

def apply_selection(events,
                    mass_window=None,
                    ptcone_max=None,
                    etcone_max=None,
                    require_both=True):
    if events is None:
        return None

    mask = ak.Array(np.ones(len(events), dtype=bool))

    # mass window
    if mass_window is not None:
        lo, hi = mass_window
        mask = mask & (events["mass"] > lo) & (events["mass"] < hi)

    # ptcone (track iso)
    if ptcone_max is not None:
        if require_both:
            mask = mask & (events["lep_ptvarcone30"][:,0] < ptcone_max) & (events["lep_ptvarcone30"][:,1] < ptcone_max)
        else:
            mask = mask & (events["lep_ptvarcone30"][:,0] < ptcone_max)

    # etcone (calo iso)
    if etcone_max is not None:
        if require_both:
            mask = mask & (events["lep_topoetcone20"][:,0] < etcone_max) & (events["lep_topoetcone20"][:,1] < etcone_max)
        else:
            mask = mask & (events["lep_topoetcone20"][:,0] < etcone_max)

    return events[mask]

def select_plotdict(plot_dict, **kwargs):
    return {k: apply_selection(v, **kwargs) if v is not None else None for k,v in plot_dict.items()}

def yield_data(events):
    return 0.0 if events is None else float(len(events))

def yield_mc(events):
    if events is None:
        return 0.0
    # backend typically stores totalWeight for MC
    if "totalWeight" in events.fields:
        return float(ak.sum(events["totalWeight"]))
    return float(len(events))

def summarize(plot_dict):
    n_data = yield_data(plot_dict.get("Data"))
    n_sig = 0.0
    n_bkg = 0.0
    for k,v in plot_dict.items():
        if k.startswith("Signal"):
            n_sig += yield_mc(v)
        if k.startswith("Background"):
            n_bkg += yield_mc(v)
    n_mc = n_sig + n_bkg
    sb = n_sig/n_bkg if n_bkg>0 else np.inf
    dmc = n_data/n_mc if n_mc>0 else np.nan
    return n_data, n_sig, n_bkg, sb, dmc

def cutflow_table(plot_dict, mass_window, ptcone_max, etcone_max, require_both=True):
    steps = [
        ("Baseline", dict()),
        ("Mass window", dict(mass_window=mass_window)),
        ("Iso only", dict(ptcone_max=ptcone_max, etcone_max=etcone_max)),
        ("Mass + Iso", dict(mass_window=mass_window, ptcone_max=ptcone_max, etcone_max=etcone_max)),
    ]
    rows = []
    for name, kw in steps:
        dsel = select_plotdict(plot_dict, require_both=require_both, **kw)
        n_data, n_sig, n_bkg, sb, dmc = summarize(dsel)
        rows.append([name, n_data, n_sig, n_bkg, sb, dmc])
    return pd.DataFrame(rows, columns=["Step","Data","Signal","Background","S/B","Data/MC"])


# -------------------------
# 9) ISO scan (float range+step) in MASS_WINDOW
#    Metrics:
#      - OS signal efficiency (relative to "mass window only")
#      - SS data rejection   (relative to "mass window only")
# -------------------------

# reference yields inside mass window only (no iso)
os_ref = select_plotdict(plot_OS, mass_window=MASS_WINDOW, require_both=REQUIRE_BOTH_ISO)
ss_ref = select_plotdict(plot_SS, mass_window=MASS_WINDOW, require_both=REQUIRE_BOTH_ISO)

_, os_sig_ref, _, _, _ = summarize(os_ref)
ss_data_ref, _, _, _, _ = summarize(ss_ref)

pt_vals = np.arange(PTCONE_RANGE[0], PTCONE_RANGE[1] + 1e-12, PTCONE_STEP)
et_vals = np.arange(ETCONE_RANGE[0], ETCONE_RANGE[1] + 1e-12, ETCONE_STEP)

rows = []
for ptc in pt_vals:
    for etc in et_vals:
        os_sel = select_plotdict(plot_OS,
                                 mass_window=MASS_WINDOW,
                                 ptcone_max=float(ptc),
                                 etcone_max=float(etc),
                                 require_both=REQUIRE_BOTH_ISO)
        ss_sel = select_plotdict(plot_SS,
                                 mass_window=MASS_WINDOW,
                                 ptcone_max=float(ptc),
                                 etcone_max=float(etc),
                                 require_both=REQUIRE_BOTH_ISO)

        os_data, os_sig, os_bkg, os_sb, os_dmc = summarize(os_sel)
        ss_data, ss_sig, ss_bkg, ss_sb, ss_dmc = summarize(ss_sel)

        os_sig_eff = os_sig/os_sig_ref if os_sig_ref > 0 else np.nan
        ss_rej = 1.0 - (ss_data/ss_data_ref) if ss_data_ref > 0 else np.nan

        rows.append([float(ptc), float(etc), os_sig_eff, ss_rej, os_dmc, os_sb, ss_data])

df_scan = pd.DataFrame(
    rows,
    columns=["ptcone_max", "etcone_max", "OS_sig_eff", "SS_rejection", "OS_Data/MC", "OS_S/B", "SS_Data"]
)

# Keep only those that satisfy OS efficiency requirement
df_allowed = df_scan[df_scan["OS_sig_eff"] >= OS_SIG_EFF_MIN].copy()

print("Scan grid size =", len(df_scan), " | Allowed by eff >=", OS_SIG_EFF_MIN, ":", len(df_allowed))

# Pick "best" = maximize SS_rejection, tie-breaker: smaller (ptcone_max+etcone_max)
df_allowed["tightness"] = df_allowed["ptcone_max"] + df_allowed["etcone_max"]
df_allowed = df_allowed.sort_values(["SS_rejection", "tightness"], ascending=[False, True])

best = df_allowed.iloc[0] if len(df_allowed) > 0 else None
best


# -------------------------
# 10) Apply best cut, plot AFTER mass (OS & SS), and print cutflow
# -------------------------

if best is None:
    raise RuntimeError("No working point satisfies OS_SIG_EFF_MIN. Try loosening ranges or lowering OS_SIG_EFF_MIN.")

BEST_PTC = float(best["ptcone_max"])
BEST_ETC = float(best["etcone_max"])

print(f"Best working point found: ptcone<{BEST_PTC:.3f} GeV, etcone<{BEST_ETC:.3f} GeV")
print("Best metrics:", best.to_dict())

plot_OS_after = select_plotdict(
    plot_OS,
    mass_window=MASS_WINDOW,
    ptcone_max=BEST_PTC,
    etcone_max=BEST_ETC,
    require_both=REQUIRE_BOTH_ISO
)

plot_SS_after = select_plotdict(
    plot_SS,
    mass_window=MASS_WINDOW,
    ptcone_max=BEST_PTC,
    etcone_max=BEST_ETC,
    require_both=REQUIRE_BOTH_ISO
)

# AFTER mass plots (zoom into Z region for clarity)
plot_variable = "mass"
xmin, xmax = 60, 120
num_bins = 60

fig_os_mass_after, _ = plot_stacked_hist(
    plot_OS_after, plot_variable, color_list,
    num_bins, xmin, xmax,
    f"mass [GeV] ({LEPTON} OS, after cuts: {MASS_WINDOW[0]}<m<{MASS_WINDOW[1]}, ptcone<{BEST_PTC:.3f}, etcone<{BEST_ETC:.3f})",
    logy=False, show_text=True, residual_plot=True, save_fig=False
)

fig_ss_mass_after, _ = plot_stacked_hist(
    plot_SS_after, plot_variable, color_list,
    num_bins, xmin, xmax,
    f"mass [GeV] ({LEPTON} SS, after cuts: {MASS_WINDOW[0]}<m<{MASS_WINDOW[1]}, ptcone<{BEST_PTC:.3f}, etcone<{BEST_ETC:.3f})",
    logy=False, show_text=True, residual_plot=True, save_fig=False
)

# Print analysis summary (before vs after) with cutflow tables
df_os_cutflow = cutflow_table(plot_OS, MASS_WINDOW, BEST_PTC, BEST_ETC, require_both=REQUIRE_BOTH_ISO)
df_ss_cutflow = cutflow_table(plot_SS, MASS_WINDOW, BEST_PTC, BEST_ETC, require_both=REQUIRE_BOTH_ISO)

print("\n=== OS cutflow ===")
display(df_os_cutflow)

print("\n=== SS cutflow ===")
display(df_ss_cutflow)
