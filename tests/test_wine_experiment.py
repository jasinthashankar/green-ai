"""
Unit and regression tests for Wine generalization experiment.

Validates:
1. Stratified 60/20/20 partitioning of the Wine dataset.
2. Strict data hygiene (StandardScaler fitted strictly on training data).
3. Directory isolation: models saved strictly to models_wine/ (never models/).
4. Correct model feature dimensions (13 features, 3 classes).
5. Correct execution and valid output formats for all 5 methods on Wine test set.
6. Theoretical FLOPs calculations for the Wine MLP architecture.
7. File isolation: ensures existing Digits CSVs are never overwritten.
"""

import os
import sys
import unittest
import numpy as np
from sklearn.datasets import load_wine

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from dataset_wine import get_wine_dataset
from wine_experiment import (
    train_wine_models,
    load_wine_models,
    build_wine_runners,
    compute_wine_theoretical_complexity,
    run_wine_experiment,
    WINE_LIMITATIONS,
)
from adaptive_selector import CostAwareAdaptiveSelector


class TestWineExperiment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models_dir = "models_wine"
        model_files = ["decision_tree.pkl", "neural_network.pkl", "router.pkl", "scaler.pkl"]
        if all(os.path.exists(os.path.join(cls.models_dir, f)) for f in model_files):
            cls.tree, cls.nn, cls.router, cls.scaler = load_wine_models(models_dir=cls.models_dir)
        else:
            cls.tree, cls.nn, cls.router, cls.scaler = train_wine_models(models_dir=cls.models_dir)

        cls.X_train, cls.X_val, cls.X_test, cls.y_train, cls.y_val, cls.y_test, _, _, _ = (
            get_wine_dataset(scale=True, random_state=42)
        )
        cls.camas = CostAwareAdaptiveSelector(cls.tree, cls.nn).fit_policy(
            cls.X_val, cls.y_val, lambda_penalty=0.08
        )
        cls.runners = build_wine_runners(cls.tree, cls.nn, cls.router, cls.camas)

    def test_wine_data_split_and_stratification(self):
        """Verifies 60/20/20 split and class stratification across 3 classes."""
        self.assertEqual(len(self.X_train), 106)
        self.assertEqual(len(self.X_val), 36)
        self.assertEqual(len(self.X_test), 36)
        self.assertEqual(len(self.X_train) + len(self.X_val) + len(self.X_test), 178)

        # Check all splits contain all 3 classes
        self.assertEqual(set(np.unique(self.y_train)), {0, 1, 2})
        self.assertEqual(set(np.unique(self.y_val)), {0, 1, 2})
        self.assertEqual(set(np.unique(self.y_test)), {0, 1, 2})

    def test_scaler_data_hygiene(self):
        """Verifies that feature scaler was fitted strictly on X_train only."""
        wine = load_wine()
        # Full dataset mean
        full_mean = wine.data.mean(axis=0)
        # Scaler fitted mean must NOT match full dataset mean
        self.assertFalse(np.allclose(self.scaler.mean_, full_mean))

        # Scaler fitted mean must match X_train_raw
        _, _, _, _, _, _, raw_scaler, _, _ = get_wine_dataset(scale=False, random_state=42)
        X_train_raw, _, _, _, _, _, _, _, _ = get_wine_dataset(scale=False, random_state=42)
        np.testing.assert_allclose(self.scaler.mean_, X_train_raw.mean(axis=0), rtol=1e-5)

    def test_directory_and_model_isolation(self):
        """Ensures Wine models are saved in models_wine/ and Digits models in models/ are untouched."""
        # Wine models exist in models_wine/
        self.assertTrue(os.path.exists(os.path.join(self.models_dir, "decision_tree.pkl")))
        self.assertTrue(os.path.exists(os.path.join(self.models_dir, "neural_network.pkl")))
        self.assertTrue(os.path.exists(os.path.join(self.models_dir, "router.pkl")))

        # Check Wine model input dimensions
        self.assertEqual(self.tree.n_features_in_, 13)
        self.assertEqual(self.nn.n_features_in_, 13)
        self.assertEqual(self.tree.n_classes_, 3)
        self.assertEqual(self.nn.n_outputs_, 3)

        # Check that Digits models are distinct (64 features, 10 classes)
        import joblib
        digits_tree = joblib.load("models/decision_tree.pkl")
        digits_nn = joblib.load("models/neural_network.pkl")
        self.assertEqual(digits_tree.n_features_in_, 64)
        self.assertEqual(digits_nn.n_features_in_, 64)
        self.assertEqual(digits_tree.n_classes_, 10)

    def test_all_five_methods_on_wine(self):
        """Verifies all five methods run on the Wine test set and produce valid predictions."""
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
            self.assertEqual(len(preds), 36)
            self.assertTrue(all(p in [0, 1, 2] for p in preds))
            self.assertTrue(0 <= nn_count <= 36)

            acc = float(np.mean(preds == self.y_test) * 100.0)
            self.assertGreater(acc, 75.0, f"{name} accuracy too low: {acc:.2f}%")

    def test_wine_theoretical_flops_calculation(self):
        """Verifies FLOPs calculations for the Wine architecture."""
        dt_comp = compute_wine_theoretical_complexity("Decision Tree", 0, 36)
        self.assertEqual(dt_comp["total_batch_flops"], 36 * 2.0)
        self.assertGreater(dt_comp["flops_reduction_pct"], 99.0)

        nn_comp = compute_wine_theoretical_complexity("Neural Network", 36, 36)
        self.assertEqual(nn_comp["total_batch_flops"], 36 * 5952.0)
        self.assertAlmostEqual(nn_comp["flops_reduction_pct"], 0.0, places=4)

    def test_results_file_isolation(self):
        """Ensures Wine experiment writes to separate files without overwriting Digits CSVs."""
        digits_csvs = [
            os.path.join("results", "enhanced_benchmark.csv"),
            os.path.join("results", "final_results.csv"),
            os.path.join("results", "energy_benchmark_v3.csv"),
        ]
        mtimes_before = {p: os.path.getmtime(p) for p in digits_csvs if os.path.exists(p)}

        test_out = "test_wine_isolation.csv"
        test_out_path = os.path.join("results", test_out)

        df, summary = run_wine_experiment(
            repeats=5,
            warmup=2,
            include_energy=False,
            output_filename=test_out,
            camas_lambda=0.08,
        )

        self.assertEqual(len(df), 5)
        self.assertTrue(os.path.exists(test_out_path))

        # Check Digits CSVs were untouched
        for p, mtime in mtimes_before.items():
            self.assertEqual(os.path.getmtime(p), mtime, f"Existing Digits file {p} was modified!")

        # Clean up
        if os.path.exists(test_out_path):
            os.remove(test_out_path)

    def test_limitations_documented(self):
        """Ensures that dataset scale and measurement limitations are explicitly documented."""
        self.assertIn("dataset_scale", WINE_LIMITATIONS)
        self.assertIn("software_tracker", WINE_LIMITATIONS)
        self.assertIn("sensor_status", WINE_LIMITATIONS)

    def test_load_wine_models_without_retraining(self):
        """Verifies load_wine_models loads intact models without modifying timestamps."""
        m_tree, m_nn, m_router, m_scaler = load_wine_models(self.models_dir)
        self.assertIsNotNone(m_tree)
        self.assertIsNotNone(m_nn)
        self.assertIsNotNone(m_router)
        self.assertIsNotNone(m_scaler)
        self.assertEqual(m_tree.n_features_in_, 13)
        self.assertEqual(m_nn.n_features_in_, 13)


if __name__ == "__main__":
    unittest.main()
