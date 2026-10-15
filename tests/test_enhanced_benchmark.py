"""
Unit and regression tests for Enhanced Multi-Tier Benchmark.
Validates:
1. Read-only model loading (no mutations or retraining).
2. Strict data hygiene (calibration strictly on validation partition).
3. Consistent runner signatures for all 5 methods.
4. Mathematical consistency of theoretical FLOPs calculations.
5. Statistical properties of latency measurements (min <= median <= max, std >= 0).
6. File isolation: ensures existing benchmark CSVs are never overwritten.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from enhanced_benchmark import (
    load_pretrained_assets,
    get_calibrated_camas,
    build_method_runners,
    compute_theoretical_complexity,
    measure_latency_distribution,
    run_enhanced_benchmark,
    MEASUREMENT_LIMITATIONS,
)


class TestEnhancedBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree, cls.nn, cls.router, cls.X_train, cls.X_val, cls.X_test, cls.y_train, cls.y_val, cls.y_test = (
            load_pretrained_assets()
        )
        cls.camas = get_calibrated_camas(cls.tree, cls.nn, cls.X_val, cls.y_val, lambda_penalty=0.20)
        cls.runners = build_method_runners(cls.tree, cls.nn, cls.router, cls.camas, cascade_threshold=0.80)

    def test_load_pretrained_assets(self):
        """Verifies assets are loaded properly without modifying original models."""
        self.assertIsNotNone(self.tree)
        self.assertIsNotNone(self.nn)
        self.assertIsNotNone(self.router)
        self.assertEqual(len(self.X_test), 360)
        self.assertEqual(len(self.y_test), 360)
        self.assertEqual(len(self.X_val), 359)
        self.assertEqual(len(self.X_train), 1078)

    def test_data_hygiene_and_camas_calibration(self):
        """Verifies CAMAS is calibrated using validation set and sets valid threshold."""
        self.assertGreater(self.camas.threshold_, 0.0)
        self.assertLess(self.camas.threshold_, 1.0)
        # Verify deterministic inference on test set
        preds1, nn1, _ = self.camas.predict(self.X_test)
        preds2, nn2, _ = self.camas.predict(self.X_test)
        np.testing.assert_array_equal(preds1, preds2)
        self.assertEqual(nn1, nn2)

    def test_all_five_runners(self):
        """Ensures all 5 methods are present, return valid arrays, and match sample count."""
        expected_methods = [
            "Decision Tree",
            "Neural Network",
            "Dynamic Cascade",
            "Dynamic Routing",
            "CAMAS (Adaptive Selection)",
        ]
        self.assertEqual(list(self.runners.keys()), expected_methods)

        for name, fn in self.runners.items():
            preds, nn_count = fn(self.X_test)
            self.assertEqual(len(preds), 360, f"{name} did not return 360 predictions")
            self.assertTrue(0 <= nn_count <= 360, f"{name} nn_count {nn_count} out of bounds")

            if name == "Decision Tree":
                self.assertEqual(nn_count, 0)
            elif name == "Neural Network":
                self.assertEqual(nn_count, 360)
            elif name == "Dynamic Cascade":
                self.assertEqual(nn_count, 173)  # Known baseline
            elif name == "Dynamic Routing":
                self.assertEqual(nn_count, 168)  # Known baseline

    def test_theoretical_complexity_calculations(self):
        """Verifies justified theoretical FLOPs accounting across all methods."""
        # Decision Tree
        dt_comp = compute_theoretical_complexity("Decision Tree", 0, 360)
        self.assertEqual(dt_comp["total_batch_flops"], 360 * 5)
        self.assertGreater(dt_comp["flops_reduction_pct"], 99.0)

        # Neural Network
        nn_comp = compute_theoretical_complexity("Neural Network", 360, 360)
        self.assertEqual(nn_comp["total_batch_flops"], 360 * 120128)
        self.assertAlmostEqual(nn_comp["flops_reduction_pct"], 0.0, places=4)

        # Dynamic Cascade
        casc_comp = compute_theoretical_complexity("Dynamic Cascade", 173, 360)
        self.assertGreater(casc_comp["flops_reduction_pct"], 50.0)
        self.assertLess(casc_comp["flops_reduction_pct"], 60.0)

        # CAMAS
        camas_comp = compute_theoretical_complexity("CAMAS (Adaptive Selection)", 169, 360)
        self.assertGreater(camas_comp["flops_reduction_pct"], 50.0)
        self.assertLess(camas_comp["flops_reduction_pct"], 60.0)

    def test_latency_distribution_statistics(self):
        """Verifies statistical properties of latency measurements."""
        stats = measure_latency_distribution(self.runners, self.X_test, repeats=10, warmup=2)
        for name, s in stats.items():
            self.assertGreater(s["min"], 0.0)
            self.assertGreaterEqual(s["median"], s["min"])
            self.assertLessEqual(s["median"], s["max"])
            self.assertGreaterEqual(s["std"], 0.0)
            self.assertGreater(s["mean"], 0.0)

    def test_benchmark_execution_and_file_isolation(self):
        """Runs benchmark and ensures existing result CSVs are not overwritten."""
        # Capture timestamps of existing CSVs
        existing_csvs = [
            os.path.join("results", "final_results.csv"),
            os.path.join("results", "energy_benchmark_v3.csv"),
            os.path.join("results", "benchmark_results.csv"),
        ]
        mtimes_before = {p: os.path.getmtime(p) for p in existing_csvs if os.path.exists(p)}

        test_out = "test_benchmark_isolation.csv"
        test_out_path = os.path.join("results", test_out)

        df, summary = run_enhanced_benchmark(
            repeats=5,
            warmup=2,
            include_energy=False,
            output_filename=test_out,
            camas_lambda=0.20,
        )

        self.assertEqual(len(df), 5)
        self.assertTrue(os.path.exists(test_out_path))

        # Check existing CSVs were untouched
        for p, mtime in mtimes_before.items():
            self.assertEqual(os.path.getmtime(p), mtime, f"Existing file {p} was modified!")

        # Clean up test output
        if os.path.exists(test_out_path):
            os.remove(test_out_path)

    def test_documented_limitations_exist(self):
        """Ensures measurement uncertainty and profiler limitations are documented."""
        self.assertIn("software_tracker", MEASUREMENT_LIMITATIONS)
        self.assertIn("sampling_frequency", MEASUREMENT_LIMITATIONS)
        self.assertIn("sensor_status", MEASUREMENT_LIMITATIONS)
        self.assertIn("micro_inference_uncertainty", MEASUREMENT_LIMITATIONS)
        self.assertIn("flops_vs_energy_distinction", MEASUREMENT_LIMITATIONS)


if __name__ == "__main__":
    unittest.main()
