from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from systematics import evaluate_isolation_neighbour_systematic  # noqa: E402


class SystematicPipelineTests(unittest.TestCase):
    def test_isolation_neighbour_systematic_is_computed_and_saved(self) -> None:
        neighbour_table = pd.DataFrame(
            [
                {"ptcone_max": 4.0, "etcone_max": 9.25, "sigma_pb": 1.20, "sigma_shift_from_nominal_pb": 0.05},
                {"ptcone_max": 5.0, "etcone_max": 9.25, "sigma_pb": 1.05, "sigma_shift_from_nominal_pb": -0.10},
                {"ptcone_max": 4.5, "etcone_max": 8.25, "sigma_pb": 1.18, "sigma_shift_from_nominal_pb": 0.03},
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "neighbours.csv"
            result = evaluate_isolation_neighbour_systematic(
                neighbour_table,
                nominal_sigma_pb=1.15,
                output_path=output_path,
            )

            self.assertTrue(output_path.exists())
            self.assertAlmostEqual(result["value_pb"], 0.10)
            self.assertEqual(len(result["neighbours"]), 3)


if __name__ == "__main__":
    unittest.main()

