from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from scan import (  # noqa: E402
    build_local_stability_diagnostics,
    build_monotonicity_diagnostics,
    build_scan_diagnostics_table,
    generate_scan_points,
)


def _evaluation_stub(ptcone_max: float, etcone_max: float) -> dict:
    base = ptcone_max + etcone_max
    sigma_results_table = pd.DataFrame(
        [
            {"method": "none", "extra_bkg": 0.0, "sigma_pb": base, "sigma_shift_pb": 0.0, "sigma_valid": True, "sigma_error": ""},
            {"method": "wrong_flavour", "extra_bkg": 1.0, "sigma_pb": base + 0.1, "sigma_shift_pb": 0.1, "sigma_valid": True, "sigma_error": ""},
            {"method": "wrong_charge", "extra_bkg": 2.0, "sigma_pb": base + 0.2, "sigma_shift_pb": 0.2, "sigma_valid": True, "sigma_error": ""},
            {"method": "both_average", "extra_bkg": 1.5, "sigma_pb": base + 0.15, "sigma_shift_pb": 0.15, "sigma_valid": True, "sigma_error": ""},
        ]
    )
    return {"sigma_results_table": sigma_results_table}


class ScanDiagnosticTests(unittest.TestCase):
    def test_generate_scan_points_local_box_includes_nominal_point(self) -> None:
        points = generate_scan_points(
            nominal_ptcone=4.5,
            nominal_etcone=9.25,
            scan_mode="local_box",
            ptcone_range=(0.0, 10.0),
            ptcone_step=1.0,
            etcone_range=(0.0, 20.0),
            etcone_step=1.0,
            local_box_ptcone_half_width=1.0,
            local_box_etcone_half_width=1.0,
        )

        self.assertIn((4.5, 9.25), points)
        self.assertIn((3.5, 8.25), points)
        self.assertIn((5.5, 10.25), points)

    def test_scan_diagnostics_table_contains_nominal_relative_shift_columns(self) -> None:
        scan_table = pd.DataFrame(
            [
                {"ptcone_max": 1.0, "etcone_max": 1.0, "OS_sig_eff": 0.99, "SS_rejection": 0.1},
                {"ptcone_max": 2.0, "etcone_max": 1.0, "OS_sig_eff": 0.98, "SS_rejection": 0.2},
            ]
        )
        nominal_sigma_lookup = pd.DataFrame(
            [
                {"method": "none", "sigma_pb": 2.0},
                {"method": "wrong_flavour", "sigma_pb": 2.1},
                {"method": "wrong_charge", "sigma_pb": 2.2},
                {"method": "both_average", "sigma_pb": 2.15},
            ]
        ).set_index("method")

        diagnostics = build_scan_diagnostics_table(
            scan_table,
            _evaluation_stub,
            nominal_sigma_lookup=nominal_sigma_lookup,
        )
        expected_columns = {
            "sigma_pb_none",
            "sigma_pb_wrong_flavour",
            "sigma_pb_wrong_charge",
            "sigma_pb_both_average",
            "sigma_shift_pb_none",
            "sigma_abs_shift_pb_wrong_charge",
            "sigma_frac_shift_both_average",
            "extra_bkg_wrong_flavour",
        }
        self.assertTrue(expected_columns.issubset(set(diagnostics.columns)))
        self.assertEqual(float(diagnostics.loc[0, "sigma_shift_pb_none"]), 0.0)
        self.assertGreater(float(diagnostics.loc[1, "sigma_abs_shift_pb_wrong_charge"]), 0.0)

    def test_monotonicity_diagnostics_mark_monotonic_surface(self) -> None:
        rows = []
        for etcone in (1.0, 2.0):
            for ptcone in (1.0, 2.0, 3.0):
                base = ptcone + 2.0 * etcone
                rows.append(
                    {
                        "ptcone_max": ptcone,
                        "etcone_max": etcone,
                        "sigma_pb_none": base,
                        "sigma_pb_wrong_flavour": base + 0.1,
                        "sigma_pb_wrong_charge": base + 0.2,
                        "sigma_pb_both_average": base + 0.15,
                    }
                )
        diagnostic_table = pd.DataFrame(rows)

        monotonicity = build_monotonicity_diagnostics(diagnostic_table, tolerance=1e-9)
        classifications = monotonicity["classification_table"]["classification"].unique().tolist()
        self.assertEqual(classifications, ["monotonic increasing"])

    def test_local_stability_uses_nominal_relative_columns(self) -> None:
        diagnostic_table = pd.DataFrame(
            [
                {
                    "ptcone_max": 4.5,
                    "etcone_max": 9.25,
                    "distance_to_nominal": 0.0,
                    "sigma_shift_pb_none": 0.0,
                    "sigma_abs_shift_pb_none": 0.0,
                    "sigma_frac_shift_none": 0.0,
                },
                {
                    "ptcone_max": 5.5,
                    "etcone_max": 9.25,
                    "distance_to_nominal": 1.0,
                    "sigma_shift_pb_none": 0.2,
                    "sigma_abs_shift_pb_none": 0.2,
                    "sigma_frac_shift_none": 0.1,
                },
            ]
        )

        local_stability = build_local_stability_diagnostics(
            diagnostic_table,
            nominal_ptcone=4.5,
            nominal_etcone=9.25,
            methods=("none",),
            max_neighbours=1,
        )
        self.assertEqual(int(local_stability.iloc[0]["neighbour_rank"]), 1)
        self.assertAlmostEqual(float(local_stability.iloc[0]["abs_slope_pb_per_gev"]), 0.2)


if __name__ == "__main__":
    unittest.main()
