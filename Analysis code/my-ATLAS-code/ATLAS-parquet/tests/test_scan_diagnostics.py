from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from scan import build_monotonicity_diagnostics, build_scan_diagnostics_table  # noqa: E402


def _evaluation_stub(ptcone_max: float, etcone_max: float) -> dict:
    base = ptcone_max + etcone_max
    sigma_results_table = pd.DataFrame(
        [
            {"method": "none", "extra_bkg": 0.0, "sigma_pb": base, "sigma_shift_pb": 0.0},
            {"method": "wrong_flavour", "extra_bkg": 1.0, "sigma_pb": base + 0.1, "sigma_shift_pb": 0.1},
            {"method": "wrong_charge", "extra_bkg": 2.0, "sigma_pb": base + 0.2, "sigma_shift_pb": 0.2},
            {"method": "both_average", "extra_bkg": 1.5, "sigma_pb": base + 0.15, "sigma_shift_pb": 0.15},
        ]
    )
    return {"sigma_results_table": sigma_results_table}


class ScanDiagnosticTests(unittest.TestCase):
    def test_scan_diagnostics_table_contains_expected_columns(self) -> None:
        scan_table = pd.DataFrame(
            [
                {"ptcone_max": 1.0, "etcone_max": 1.0, "significance": 3.0, "OS_sig_eff": 0.99, "SS_rejection": 0.1},
                {"ptcone_max": 2.0, "etcone_max": 1.0, "significance": 3.5, "OS_sig_eff": 0.98, "SS_rejection": 0.2},
            ]
        )

        diagnostics = build_scan_diagnostics_table(scan_table, _evaluation_stub)
        expected_columns = {
            "significance",
            "sigma_pb_none",
            "sigma_pb_wrong_flavour",
            "sigma_pb_wrong_charge",
            "sigma_pb_both_average",
            "extra_bkg_wrong_flavour",
            "extra_bkg_wrong_charge",
            "extra_bkg_both_average",
        }
        self.assertTrue(expected_columns.issubset(set(diagnostics.columns)))

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


if __name__ == "__main__":
    unittest.main()

