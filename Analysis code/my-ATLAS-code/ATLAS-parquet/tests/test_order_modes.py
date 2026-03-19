from __future__ import annotations

import sys
import unittest
from pathlib import Path

import awkward as ak


ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from config import CHANNELS  # noqa: E402
from control_regions import evaluate_sigma_with_estimator  # noqa: E402


def _make_events(count: int, weight: float = 1.0) -> ak.Array:
    return ak.Array(
        {
            "mass": [91.0] * count,
            "lep_ptvarcone30": [[1.0, 1.0]] * count,
            "lep_topoetcone20": [[1.0, 1.0]] * count,
            "weight": [weight] * count,
        }
    )


def _produced_event_count_stub(sample_key: str, lumi_fb: float) -> None:
    print(f"{sample_key} {lumi_fb} 100")


class OrderModeTests(unittest.TestCase):
    def test_recompute_after_iso_and_fixed_before_iso_can_differ(self) -> None:
        plot_os = {
            "Data": _make_events(20, weight=1.0),
            "Signal Zmumu": _make_events(50, weight=1.0),
            "Signal (LowMassDY m10_40_Zmumu)": _make_events(1, weight=0.0),
            "Background Ztautau": _make_events(2, weight=1.0),
            "Background ttbar": _make_events(1, weight=1.0),
            "Background Wmunu": _make_events(1, weight=1.0),
        }
        stage_totals = {
            "baseline": {
                "data_counts": {
                    "ordered_11_13": 12.0,
                    "ordered_13_11": 0.0,
                    "ep_mum": 6.0,
                    "mup_em": 4.0,
                    "ee_ss": 0.0,
                    "mumu_ss": 5.0,
                },
                "mc_counts": {
                    "ordered_11_13": 3.0,
                    "ordered_13_11": 0.0,
                    "ep_mum": 2.0,
                    "mup_em": 1.0,
                    "ee_ss": 0.0,
                    "mumu_ss": 2.0,
                },
                "mc_samples_included": [],
            },
            "mass_only": {
                "data_counts": {
                    "ordered_11_13": 12.0,
                    "ordered_13_11": 0.0,
                    "ep_mum": 6.0,
                    "mup_em": 4.0,
                    "ee_ss": 0.0,
                    "mumu_ss": 5.0,
                },
                "mc_counts": {
                    "ordered_11_13": 3.0,
                    "ordered_13_11": 0.0,
                    "ep_mum": 2.0,
                    "mup_em": 1.0,
                    "ee_ss": 0.0,
                    "mumu_ss": 2.0,
                },
                "mc_samples_included": [],
            },
            "mass_plus_iso": {
                "data_counts": {
                    "ordered_11_13": 5.0,
                    "ordered_13_11": 0.0,
                    "ep_mum": 2.0,
                    "mup_em": 2.0,
                    "ee_ss": 0.0,
                    "mumu_ss": 2.0,
                },
                "mc_counts": {
                    "ordered_11_13": 2.0,
                    "ordered_13_11": 0.0,
                    "ep_mum": 1.0,
                    "mup_em": 1.0,
                    "ee_ss": 0.0,
                    "mumu_ss": 1.0,
                },
                "mc_samples_included": [],
            },
        }

        fixed_before_iso = evaluate_sigma_with_estimator(
            lepton="mu",
            plot_os=plot_os,
            channel_config=CHANNELS["mu"],
            produced_event_count_fn=_produced_event_count_stub,
            backend={},
            mass_window=(66.0, 116.0),
            ptcone_max=4.5,
            etcone_max=9.25,
            require_both=True,
            method="both_average",
            order_mode="fixed_before_iso",
            stage_totals=stage_totals,
        )
        recompute_after_iso = evaluate_sigma_with_estimator(
            lepton="mu",
            plot_os=plot_os,
            channel_config=CHANNELS["mu"],
            produced_event_count_fn=_produced_event_count_stub,
            backend={},
            mass_window=(66.0, 116.0),
            ptcone_max=4.5,
            etcone_max=9.25,
            require_both=True,
            method="both_average",
            order_mode="recompute_after_iso",
            stage_totals=stage_totals,
        )

        self.assertNotEqual(
            fixed_before_iso["applied_extra_background"],
            recompute_after_iso["applied_extra_background"],
        )
        self.assertNotEqual(
            fixed_before_iso["sigma_with_additional_bkg"]["sigma_pb"],
            recompute_after_iso["sigma_with_additional_bkg"]["sigma_pb"],
        )


if __name__ == "__main__":
    unittest.main()
