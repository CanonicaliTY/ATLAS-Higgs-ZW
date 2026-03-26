from __future__ import annotations

import sys
import unittest
from pathlib import Path

import awkward as ak


ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from config import CHANNELS  # noqa: E402
from control_regions import evaluate_cut_point, serialise_cut_point_result  # noqa: E402


def _make_events(ptcone_values: list[float], weight: float = 1.0) -> ak.Array:
    return ak.Array(
        {
            "mass": [91.0] * len(ptcone_values),
            "lep_ptvarcone30": [[value, value] for value in ptcone_values],
            "lep_topoetcone20": [[value, value] for value in ptcone_values],
            "weight": [weight] * len(ptcone_values),
        }
    )


def _produced_event_count_stub(sample_key: str, lumi_fb: float) -> None:
    print(f"{sample_key} {lumi_fb} 100")


def _stage_totals(ep_mum: float, mup_em: float, mumu_ss: float) -> dict:
    return {
        "baseline": {
            "data_counts": {
                "ordered_11_13": 6.0,
                "ordered_13_11": 0.0,
                "ep_mum": 3.0,
                "mup_em": 3.0,
                "ee_ss": 0.0,
                "mumu_ss": 3.0,
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
        "mass_only": {
            "data_counts": {
                "ordered_11_13": 6.0,
                "ordered_13_11": 0.0,
                "ep_mum": 3.0,
                "mup_em": 3.0,
                "ee_ss": 0.0,
                "mumu_ss": 3.0,
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
        "mass_plus_iso": {
            "data_counts": {
                "ordered_11_13": ep_mum + mup_em,
                "ordered_13_11": 0.0,
                "ep_mum": ep_mum,
                "mup_em": mup_em,
                "ee_ss": 0.0,
                "mumu_ss": mumu_ss,
            },
            "mc_counts": {
                "ordered_11_13": 1.0,
                "ordered_13_11": 0.0,
                "ep_mum": 0.5,
                "mup_em": 0.5,
                "ee_ss": 0.0,
                "mumu_ss": 0.5,
            },
            "mc_samples_included": [],
        },
    }


class CutPointEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plot_os = {
            "Data": _make_events([1.0, 2.0, 5.0, 8.0], weight=1.0),
            "Signal Zmumu": _make_events([1.0, 2.0, 5.0, 8.0], weight=1.0),
            "Signal (LowMassDY m10_40_Zmumu)": _make_events([1.0], weight=0.0),
            "Background Ztautau": _make_events([1.0, 8.0], weight=1.0),
            "Background ttbar": _make_events([2.0], weight=1.0),
            "Background Wmunu": _make_events([5.0], weight=1.0),
        }

    def test_cut_point_returns_all_required_method_rows(self) -> None:
        result = evaluate_cut_point(
            lepton="mu",
            plot_os=self.plot_os,
            plot_ss=None,
            channel_config=CHANNELS["mu"],
            produced_event_count_fn=_produced_event_count_stub,
            backend=None,
            mass_window=(66.0, 116.0),
            ptcone_max=10.0,
            etcone_max=10.0,
            require_both=True,
            stage_totals=_stage_totals(3.0, 2.0, 2.0),
        )

        methods = result["sigma_results_table"]["method"].tolist()
        self.assertEqual(methods, ["none", "wrong_flavour", "wrong_charge", "both_average"])
        self.assertNotIn("both_sum", result["estimators_table"]["method"].tolist())
        self.assertIn("wrong_flavour", result["comparison_sigma_if_applied"]["method"].tolist())
        self.assertIn("wrong_charge", result["comparison_sigma_if_applied"]["method"].tolist())
        self.assertIn("both_average", result["comparison_sigma_if_applied"]["method"].tolist())

    def test_selected_method_style_outputs_are_absent(self) -> None:
        result = evaluate_cut_point(
            lepton="mu",
            plot_os=self.plot_os,
            plot_ss=None,
            channel_config=CHANNELS["mu"],
            produced_event_count_fn=_produced_event_count_stub,
            backend=None,
            mass_window=(66.0, 116.0),
            ptcone_max=10.0,
            etcone_max=10.0,
            require_both=True,
            stage_totals=_stage_totals(3.0, 2.0, 2.0),
        )
        serialised = serialise_cut_point_result(result)

        self.assertNotIn("selected_method", result)
        self.assertNotIn("selected_estimator", result)
        self.assertNotIn("significance", result["signal_region_yields"])
        self.assertNotIn("selected_method", serialised)
        self.assertNotIn("selected_estimator", serialised)

    def test_signal_region_and_control_region_change_consistently_with_cut_point(self) -> None:
        loose = evaluate_cut_point(
            lepton="mu",
            plot_os=self.plot_os,
            plot_ss=None,
            channel_config=CHANNELS["mu"],
            produced_event_count_fn=_produced_event_count_stub,
            backend=None,
            mass_window=(66.0, 116.0),
            ptcone_max=10.0,
            etcone_max=10.0,
            require_both=True,
            stage_totals=_stage_totals(3.0, 2.0, 2.0),
        )
        tight = evaluate_cut_point(
            lepton="mu",
            plot_os=self.plot_os,
            plot_ss=None,
            channel_config=CHANNELS["mu"],
            produced_event_count_fn=_produced_event_count_stub,
            backend=None,
            mass_window=(66.0, 116.0),
            ptcone_max=3.0,
            etcone_max=3.0,
            require_both=True,
            stage_totals=_stage_totals(1.0, 1.0, 0.5),
        )

        self.assertGreater(loose["signal_region_yields"]["Data"], tight["signal_region_yields"]["Data"])

        loose_sigma = loose["sigma_results_table"].set_index("method")
        tight_sigma = tight["sigma_results_table"].set_index("method")
        self.assertNotEqual(
            float(loose_sigma.loc["wrong_flavour", "extra_bkg"]),
            float(tight_sigma.loc["wrong_flavour", "extra_bkg"]),
        )
        self.assertNotEqual(
            float(loose_sigma.loc["wrong_flavour", "sigma_pb"]),
            float(tight_sigma.loc["wrong_flavour", "sigma_pb"]),
        )


if __name__ == "__main__":
    unittest.main()
