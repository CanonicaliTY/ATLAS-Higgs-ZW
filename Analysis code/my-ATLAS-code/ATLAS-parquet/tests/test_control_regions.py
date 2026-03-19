from __future__ import annotations

import sys
import unittest
from pathlib import Path

import awkward as ak


ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from control_regions import control_region_masks, summarise_control_regions  # noqa: E402


class ControlRegionMaskTests(unittest.TestCase):
    def test_physical_wrong_flavour_masks_are_order_invariant(self) -> None:
        events = ak.Array(
            {
                "lep_type": [[11, 13], [11, 13]],
                "lep_charge": [[1, -1], [-1, 1]],
                "lep_ptvarcone30": [[1.0, 1.0], [1.0, 1.0]],
                "lep_topoetcone20": [[1.0, 1.0], [1.0, 1.0]],
                "mass": [91.0, 91.0],
            }
        )

        masks = control_region_masks(events)
        summary = summarise_control_regions(events)

        self.assertEqual(int(ak.sum(masks["ep_mum"])), 1)
        self.assertEqual(int(ak.sum(masks["mup_em"])), 1)

        # The ordered flavour pairs are entirely (11,13) here, but the physical
        # wrong-flavour charge categories are both populated.
        self.assertEqual(summary["count"]["ordered_11_13"], 2.0)
        self.assertEqual(summary["count"]["ordered_13_11"], 0.0)
        self.assertEqual(summary["count"]["ep_mum"], 1.0)
        self.assertEqual(summary["count"]["mup_em"], 1.0)


if __name__ == "__main__":
    unittest.main()

