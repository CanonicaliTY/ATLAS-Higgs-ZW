from __future__ import annotations

import importlib
import importlib.util
import sys
import unittest
from pathlib import Path


ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))


class SmokeImportTests(unittest.TestCase):
    def test_modules_import_cleanly(self) -> None:
        module_names = [
            "config",
            "utils",
            "selections",
            "parquet_io",
            "control_regions",
            "scan",
            "cross_section",
            "systematics",
            "visualisation",
            "main",
        ]
        for module_name in module_names:
            with self.subTest(module=module_name):
                importlib.import_module(module_name)

    def test_legacy_wrapper_imports_main_entry_point(self) -> None:
        wrapper_path = ANALYSIS_DIR / "RearrangedCode_v2.0.py"
        spec = importlib.util.spec_from_file_location("legacy_wrapper", wrapper_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(callable(module.main))


if __name__ == "__main__":
    unittest.main()

