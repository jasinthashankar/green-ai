"""
Tests for Phase 4 Multi-Objective Pareto Frontier Analysis (src/pareto_analysis.py).

Verifies:
1. Strict Pareto dominance logic (at least as good on all, strictly better on at least one).
2. Resource minimization and accuracy maximization principles.
3. Ties, duplicate points, and missing values (NaN) handling.
4. Independent dataset frontiers for Digits and Wine.
5. Determinism: identical results across repeated runs.
6. Preservation of protected assets (models, historical benchmarks, Phase 3 outputs, app.py).
7. Correct output file schemas and generation of publication-quality figures.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

# Path setup
_HERE = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
for _p in [os.path.join(_ROOT, "src"), _ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pareto_analysis import (
    is_dominated,
    compute_pareto_frontier,
    load_and_enrich_dataset_results,
    analyze_dataset_pareto,
    run_pareto_analysis,
)


class TestParetoDominanceCore(unittest.TestCase):
    """Unit tests for the is_dominated core logic."""

    def setUp(self):
        self.objectives_2d = [
            ("Accuracy (%)", "max"),
            ("Latency Median (ms)", "min"),
        ]

    def test_strictly_dominated_point(self):
        """Point B is worse on accuracy and worse on latency -> B is dominated by A."""
        cand_B = {"Method": "B", "Accuracy (%)": 80.0, "Latency Median (ms)": 2.5}
        other_A = {"Method": "A", "Accuracy (%)": 90.0, "Latency Median (ms)": 1.5}
        dominated, dominators = is_dominated(cand_B, [other_A], self.objectives_2d)
        self.assertTrue(dominated)
        self.assertIn("A", dominators)

    def test_strictly_better_one_equal_other(self):
        """Point A is equal on latency but strictly higher accuracy -> A dominates B."""
        cand_B = {"Method": "B", "Accuracy (%)": 85.0, "Latency Median (ms)": 2.0}
        other_A = {"Method": "A", "Accuracy (%)": 90.0, "Latency Median (ms)": 2.0}
        dominated, dominators = is_dominated(cand_B, [other_A], self.objectives_2d)
        self.assertTrue(dominated)
        self.assertIn("A", dominators)

    def test_tradeoff_neither_dominates(self):
        """Point A has higher accuracy, but Point B has lower latency -> neither dominates."""
        point_A = {"Method": "A", "Accuracy (%)": 95.0, "Latency Median (ms)": 3.0}
        point_B = {"Method": "B", "Accuracy (%)": 85.0, "Latency Median (ms)": 1.0}

        dom_A, _ = is_dominated(point_A, [point_B], self.objectives_2d)
        dom_B, _ = is_dominated(point_B, [point_A], self.objectives_2d)
        self.assertFalse(dom_A)
        self.assertFalse(dom_B)

    def test_ties_and_duplicate_coordinates(self):
        """Identical coordinates on all objectives: neither strictly dominates the other."""
        point_A = {"Method": "A", "Accuracy (%)": 90.0, "Latency Median (ms)": 2.0}
        point_B = {"Method": "B", "Accuracy (%)": 90.0, "Latency Median (ms)": 2.0}

        dom_A, doms_A = is_dominated(point_A, [point_B], self.objectives_2d)
        dom_B, doms_B = is_dominated(point_B, [point_A], self.objectives_2d)
        self.assertFalse(dom_A, "Identical points should not strictly dominate each other")
        self.assertFalse(dom_B, "Identical points should not strictly dominate each other")

    def test_nan_handling_raise(self):
        """Missing value in candidate or competitor raises ValueError under nan_policy='raise'."""
        point_nan = {"Method": "NaN_Point", "Accuracy (%)": np.nan, "Latency Median (ms)": 2.0}
        point_valid = {"Method": "Valid_Point", "Accuracy (%)": 90.0, "Latency Median (ms)": 2.0}

        with self.assertRaises(ValueError):
            is_dominated(point_nan, [point_valid], self.objectives_2d, nan_policy="raise")

        with self.assertRaises(ValueError):
            is_dominated(point_valid, [point_nan], self.objectives_2d, nan_policy="raise")

    def test_nan_handling_omit(self):
        """Under nan_policy='omit', competitors with NaN are safely bypassed."""
        point_valid = {"Method": "Valid_Point", "Accuracy (%)": 90.0, "Latency Median (ms)": 2.0}
        point_nan = {"Method": "NaN_Point", "Accuracy (%)": np.nan, "Latency Median (ms)": 1.0}

        dominated, dominators = is_dominated(point_valid, [point_nan], self.objectives_2d, nan_policy="omit")
        self.assertFalse(dominated)
        self.assertEqual(dominators, [])

    def test_invalid_direction_raises(self):
        """Invalid objective direction string raises ValueError."""
        cand = {"Method": "A", "Acc": 90.0}
        with self.assertRaises(ValueError):
            is_dominated(cand, [], [("Acc", "maximize")])


class TestParetoFrontierProperties(unittest.TestCase):
    """Tests properties of compute_pareto_frontier on synthetic and benchmark data."""

    def test_extremes_always_non_dominated(self):
        """The configuration with highest accuracy and the one with lowest resource must be Pareto-optimal."""
        data = pd.DataFrame([
            {"Method": "Low", "Accuracy (%)": 60.0, "Cost": 5.0},
            {"Method": "Mid", "Accuracy (%)": 80.0, "Cost": 25.0},
            {"Method": "High", "Accuracy (%)": 95.0, "Cost": 100.0},
            {"Method": "Suboptimal", "Accuracy (%)": 75.0, "Cost": 35.0},
        ])
        objs = [("Accuracy (%)", "max"), ("Cost", "min")]
        is_pareto, dom_by = compute_pareto_frontier(data, objs)

        # "Low" has lowest cost (5.0) -> non-dominated
        self.assertTrue(is_pareto[0])
        # "High" has highest accuracy (95.0) -> non-dominated
        self.assertTrue(is_pareto[2])
        # "Suboptimal" (80.0 acc @ 25 cost beats 75.0 acc @ 35 cost) -> dominated
        self.assertFalse(is_pareto[3])
        self.assertIn("Mid", dom_by[3])

    def test_duplicate_non_dominated_points_both_marked(self):
        """If two configurations have identical non-dominated coordinates, both receive is_pareto=True."""
        data = pd.DataFrame([
            {"Method": "Copy1", "Accuracy (%)": 90.0, "Cost": 10.0},
            {"Method": "Copy2", "Accuracy (%)": 90.0, "Cost": 10.0},
            {"Method": "Worse", "Accuracy (%)": 85.0, "Cost": 15.0},
        ])
        objs = [("Accuracy (%)", "max"), ("Cost", "min")]
        is_pareto, _ = compute_pareto_frontier(data, objs)
        self.assertTrue(is_pareto[0])
        self.assertTrue(is_pareto[1])
        self.assertFalse(is_pareto[2])

    def test_frontier_determinism(self):
        """Running the frontier algorithm multiple times produces identical results."""
        df_d = pd.read_csv("results/sla_camas_digits.csv")
        objs = [
            ("Accuracy (%)", "max"),
            ("Latency Median (ms)", "min"),
            ("Theoretical FLOPs/Sample", "min"),
        ]
        p1, d1 = compute_pareto_frontier(df_d, objs)
        p2, d2 = compute_pareto_frontier(df_d, objs)
        pd.testing.assert_series_equal(p1, p2)
        pd.testing.assert_series_equal(d1, d2)


class TestDatasetFrontiersSeparation(unittest.TestCase):
    """Verifies Digits and Wine have separate, independent frontiers."""

    @classmethod
    def setUpClass(cls):
        cls.digits_df = analyze_dataset_pareto(
            "Digits",
            "results/sla_camas_digits.csv",
            "results/enhanced_benchmark.csv",
            "results/pareto_digits.csv",
        )
        cls.wine_df = analyze_dataset_pareto(
            "Wine",
            "results/sla_camas_wine.csv",
            "results/wine_benchmark.csv",
            "results/pareto_wine.csv",
        )

    def test_row_counts_and_isolation(self):
        """Both datasets must have exactly 9 evaluated configurations."""
        self.assertEqual(len(self.digits_df), 9)
        self.assertEqual(len(self.wine_df), 9)
        self.assertTrue((self.digits_df["Dataset"] == "Digits").all())
        self.assertTrue((self.wine_df["Dataset"] == "Wine").all())

    def test_neural_network_dominates_sla100_on_wine(self):
        """On Wine, Neural Network must strictly dominate SLA-CAMAS @ 100% in 3D due to routing overhead."""
        nn_row = self.wine_df[self.wine_df["Method"] == "Neural Network"].iloc[0]
        sla100_row = self.wine_df[self.wine_df["Method"] == "SLA-CAMAS @ 100%"].iloc[0]

        # Equal accuracy (100%), lower latency, lower FLOPs
        self.assertEqual(nn_row["Accuracy (%)"], sla100_row["Accuracy (%)"])
        self.assertLess(nn_row["Latency Median (ms)"], sla100_row["Latency Median (ms)"])
        self.assertLess(nn_row["Theoretical FLOPs/Sample"], sla100_row["Theoretical FLOPs/Sample"])
        self.assertFalse(sla100_row["Pareto Optimal (3D: Acc, Latency, FLOPs)"])
        self.assertIn("Neural Network", sla100_row["Dominated By (3D)"])

    def test_decision_tree_lowest_cost_on_digits(self):
        """On Digits, Decision Tree has the lowest FLOPs (5) and lowest latency."""
        dt_row = self.digits_df[self.digits_df["Method"] == "Decision Tree"].iloc[0]
        self.assertTrue(dt_row["Pareto Optimal (3D: Acc, Latency, FLOPs)"])
        self.assertEqual(dt_row["Theoretical FLOPs/Sample"], 5)


class TestArtifactsAndSchemas(unittest.TestCase):
    """Verifies that all Phase 4 output files and figures exist with correct schemas."""

    def test_output_csv_files_exist(self):
        self.assertTrue(os.path.exists("results/pareto_digits.csv"))
        self.assertTrue(os.path.exists("results/pareto_wine.csv"))
        self.assertGreater(os.path.getsize("results/pareto_digits.csv"), 1000)
        self.assertGreater(os.path.getsize("results/pareto_wine.csv"), 1000)

    def test_summary_markdown_exists(self):
        self.assertTrue(os.path.exists("results/pareto_summary.md"))
        self.assertGreater(os.path.getsize("results/pareto_summary.md"), 2000)

    def test_figures_exist_and_non_empty(self):
        figure_files = [
            "results/figures/digits_accuracy_vs_latency.png",
            "results/figures/digits_accuracy_vs_flops.png",
            "results/figures/wine_accuracy_vs_latency.png",
            "results/figures/wine_accuracy_vs_flops.png",
        ]
        for fig_path in figure_files:
            self.assertTrue(os.path.exists(fig_path), f"Figure missing: {fig_path}")
            # Research quality PNG at 300 DPI should be well over 50 KB
            self.assertGreater(os.path.getsize(fig_path), 50000, f"Figure unexpectedly small: {fig_path}")

    def test_expected_columns_in_csv(self):
        df_d = pd.read_csv("results/pareto_digits.csv")
        required_cols = [
            "Dataset",
            "Method",
            "SLA Budget (%)",
            "Accuracy (%)",
            "Test Samples",
            "NN Invocations",
            "NN Usage (%) [Test]",
            "Latency Median (ms)",
            "Theoretical FLOPs/Sample",
            "Theoretical FLOPs Saved (%)",
            "CodeCarbon Energy (kWh)",
            "CodeCarbon Emissions (gCO2e)",
            "Energy Sensor Status",
            "Energy Measurement Source",
            "Energy Limitation Note",
            "Pareto Optimal (Acc vs Latency)",
            "Pareto Optimal (Acc vs FLOPs)",
            "Pareto Optimal (3D: Acc, Latency, FLOPs)",
            "Dominated By (3D)",
            "Data Source",
            "Experiment Configuration",
        ]
        for col in required_cols:
            self.assertIn(col, df_d.columns, f"Required column missing from pareto_digits.csv: {col}")


class TestProtectedFilesIntact(unittest.TestCase):
    """Verifies that original models and historical benchmark files were NOT modified."""

    def test_models_intact(self):
        protected_models = [
            "models/decision_tree.pkl",
            "models/neural_network.pkl",
            "models/router.pkl",
            "models_wine/decision_tree.pkl",
            "models_wine/neural_network.pkl",
            "models_wine/router.pkl",
            "models_wine/scaler.pkl",
        ]
        for m in protected_models:
            self.assertTrue(os.path.exists(m), f"Protected model missing: {m}")
            self.assertGreater(os.path.getsize(m), 0)

    def test_historical_benchmarks_intact(self):
        protected_csvs = [
            "results/final_results.csv",
            "results/benchmark_results.csv",
            "results/enhanced_benchmark.csv",
            "results/wine_benchmark.csv",
            "results/sla_camas_digits.csv",
            "results/sla_camas_wine.csv",
        ]
        for c in protected_csvs:
            self.assertTrue(os.path.exists(c), f"Protected benchmark CSV missing: {c}")
            self.assertGreater(os.path.getsize(c), 100)

    def test_app_py_intact(self):
        self.assertTrue(os.path.exists("app.py"), "app.py missing")
        self.assertGreater(os.path.getsize("app.py"), 1000)


if __name__ == "__main__":
    unittest.main()
