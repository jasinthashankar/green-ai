"""
Unit and regression tests for CostAwareAdaptiveSelector (CAMAS).
Verifies mathematical properties of uncertainty metrics, calibration data hygiene,
vectorized inference determinism, and non-regression against existing models.
"""

import os
import sys
import unittest
import joblib
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from dataset import get_dataset
from adaptive_selector import (
    CostAwareAdaptiveSelector,
    compute_uncertainty_metrics,
    get_theoretical_flops,
)


class TestCAMAS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = joblib.load("models/decision_tree.pkl")
        cls.neural_net = joblib.load("models/neural_network.pkl")
        cls.router = joblib.load("models/router.pkl")
        cls.X_train, cls.X_val, cls.X_test, cls.y_train, cls.y_val, cls.y_test = get_dataset()

    def test_existing_models_preserved(self):
        """Ensure all 3 required models are loaded and intact."""
        self.assertIsNotNone(self.tree)
        self.assertIsNotNone(self.neural_net)
        self.assertIsNotNone(self.router)
        self.assertEqual(len(self.X_test), 360)
        self.assertEqual(len(self.y_test), 360)

    def test_uncertainty_metrics_boundary_conditions(self):
        """Test normalized entropy, margin, and composite score on theoretical boundaries."""
        # 1. Perfectly confident distribution: [1.0, 0, 0, ...]
        one_hot = np.zeros((1, 10))
        one_hot[0, 3] = 1.0
        metrics_one_hot = compute_uncertainty_metrics(one_hot)

        self.assertAlmostEqual(float(metrics_one_hot["confidence"][0]), 1.0, places=4)
        self.assertAlmostEqual(float(metrics_one_hot["margin"][0]), 1.0, places=4)
        self.assertAlmostEqual(float(metrics_one_hot["normalized_entropy"][0]), 0.0, places=3)
        self.assertAlmostEqual(float(metrics_one_hot["composite_uncertainty"][0]), 0.0, places=3)

        # 2. Maximum ambiguity uniform distribution: [0.1, 0.1, ...]
        uniform = np.full((1, 10), 0.1)
        metrics_uniform = compute_uncertainty_metrics(uniform)

        self.assertAlmostEqual(float(metrics_uniform["confidence"][0]), 0.1, places=4)
        self.assertAlmostEqual(float(metrics_uniform["margin"][0]), 0.0, places=4)
        self.assertAlmostEqual(float(metrics_uniform["normalized_entropy"][0]), 1.0, places=3)
        self.assertAlmostEqual(float(metrics_uniform["composite_uncertainty"][0]), 1.0, places=3)

    def test_uncertainty_metrics_real_data(self):
        """Verify metric bounds [0, 1] on real tree probabilities."""
        tree_probs = self.tree.predict_proba(self.X_val)
        metrics = compute_uncertainty_metrics(tree_probs)

        for key, arr in metrics.items():
            self.assertEqual(len(arr), len(self.X_val))
            self.assertTrue(np.all(arr >= -1e-6), f"{key} below 0")
            self.assertTrue(np.all(arr <= 1.0 + 1e-6), f"{key} above 1")

    def test_policy_fit_and_data_hygiene(self):
        """Ensure fit_policy calibrates on validation set and sets valid threshold."""
        selector = CostAwareAdaptiveSelector(self.tree, self.neural_net)
        selector.fit_policy(self.X_val, self.y_val, lambda_penalty=0.30)

        self.assertGreaterEqual(selector.threshold_, 0.0)
        self.assertLessEqual(selector.threshold_, 1.0)
        self.assertGreater(len(selector.pareto_frontier_), 0)

        profiles = selector.get_pareto_profiles()
        self.assertIn("max_accuracy", profiles)
        self.assertIn("balanced_green", profiles)
        self.assertIn("ultra_green", profiles)

    def test_inference_determinism_and_performance(self):
        """Verify that CAMAS inference is deterministic and maintains strong test accuracy."""
        selector = CostAwareAdaptiveSelector(self.tree, self.neural_net)
        selector.fit_policy(self.X_val, self.y_val, lambda_penalty=0.30)

        preds_1, nn_count_1, _ = selector.predict(self.X_test)
        preds_2, nn_count_2, _ = selector.predict(self.X_test)

        # Deterministic
        np.testing.assert_array_equal(preds_1, preds_2)
        self.assertEqual(nn_count_1, nn_count_2)

        # Quality check: test accuracy must exceed decision tree baseline (68.3%)
        acc = np.mean(preds_1 == self.y_test) * 100.0
        self.assertGreater(acc, 90.0, f"Expected accuracy > 90%, got {acc:.2f}%")

    def test_theoretical_flops_calculation(self):
        """Verify theoretical FLOPs calculation logic."""
        flops = get_theoretical_flops(n_samples=360, nn_usage_fraction=0.50)
        self.assertGreater(flops["total_batch_flops"], 0)
        self.assertGreater(flops["theoretical_flops_reduction_pct"], 40.0)
        self.assertLess(flops["theoretical_flops_reduction_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
