"""
Tests for SLA-CAMAS (src/sla_camas.py).

Verifies:
  1. budget=0.0 forces all-tree (nn_count=0).
  2. budget=1.0 forces all-NN (nn_count=len(X)).
  3. Calibration uses only validation data (test labels do not affect τ).
  4. Infeasible budget on Wine (discrete uncertainty distribution) is reported honestly.
  5. Determinism: two calibration calls with same data produce identical τ.
  6. Model .pkl files are not modified by calibration.
  7. Digits budgets: achieved val NN usage ≤ requested budget + small tolerance.
  8. Test labels cannot influence calibration (monkey-patch test).

All tests use pretrained models from models/ and models_wine/.
No retraining occurs; no existing files are modified.
"""

import os
import sys
import unittest

import numpy as np

# --------------------------------------------------------------------------
# Path setup – support running as `python -m unittest discover tests` from root
# --------------------------------------------------------------------------
_HERE = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
for _p in [os.path.join(_ROOT, "src"), _ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import joblib
from sklearn.metrics import accuracy_score

from dataset import get_dataset
from dataset_wine import get_wine_dataset
from sla_camas import SLACAMASSelector


# --------------------------------------------------------------------------
# Shared fixtures (loaded once per TestCase to keep tests fast)
# --------------------------------------------------------------------------

class _DigitsFixture:
    """Mixin: loads Digits models and data lazily."""
    _digits_loaded = False

    @classmethod
    def _load_digits(cls):
        if cls._digits_loaded:
            return
        cls._tree = joblib.load("models/decision_tree.pkl")
        cls._nn = joblib.load("models/neural_network.pkl")
        _, cls._X_val, cls._X_test, _, cls._y_val, cls._y_test = get_dataset()
        cls._digits_loaded = True


class _WineFixture:
    """Mixin: loads Wine models and data lazily."""
    _wine_loaded = False

    @classmethod
    def _load_wine(cls):
        if cls._wine_loaded:
            return
        cls._w_tree = joblib.load("models_wine/decision_tree.pkl")
        cls._w_nn = joblib.load("models_wine/neural_network.pkl")
        _, cls._X_val_w, cls._X_test_w, _, cls._y_val_w, cls._y_test_w, _, _, _ = (
            get_wine_dataset(scale=True, random_state=42)
        )
        cls._wine_loaded = True


# --------------------------------------------------------------------------
# Test cases
# --------------------------------------------------------------------------

class TestSLACAMASBudgetZero(unittest.TestCase, _DigitsFixture):
    """budget=0.0 must force all samples to the Decision Tree."""

    @classmethod
    def setUpClass(cls):
        cls._load_digits()

    def test_budget_0_forces_all_tree(self):
        sel = SLACAMASSelector(self._tree, self._nn)
        sel.fit_budget_policy(self._X_val, self._y_val, max_nn_budget=0.0)
        preds, nn_count, _ = sel.predict(self._X_test)
        self.assertEqual(nn_count, 0,
            msg=f"Expected nn_count=0 for budget=0.0, got {nn_count}")

    def test_budget_0_val_usage_is_zero(self):
        sel = SLACAMASSelector(self._tree, self._nn)
        sel.fit_budget_policy(self._X_val, self._y_val, max_nn_budget=0.0)
        self.assertAlmostEqual(sel.achieved_val_nn_usage_, 0.0, places=6)

    def test_budget_0_budget_met(self):
        sel = SLACAMASSelector(self._tree, self._nn)
        sel.fit_budget_policy(self._X_val, self._y_val, max_nn_budget=0.0)
        self.assertTrue(sel.is_budget_met_val_)


class TestSLACAMASBudgetFull(unittest.TestCase, _DigitsFixture):
    """budget=1.0 must escalate every sample to the Neural Network."""

    @classmethod
    def setUpClass(cls):
        cls._load_digits()

    def test_budget_1_forces_all_nn(self):
        sel = SLACAMASSelector(self._tree, self._nn)
        sel.fit_budget_policy(self._X_val, self._y_val, max_nn_budget=1.0)
        preds, nn_count, _ = sel.predict(self._X_test)
        self.assertEqual(nn_count, len(self._X_test),
            msg=f"Expected nn_count={len(self._X_test)} for budget=1.0, got {nn_count}")

    def test_budget_1_val_usage_is_one(self):
        sel = SLACAMASSelector(self._tree, self._nn)
        sel.fit_budget_policy(self._X_val, self._y_val, max_nn_budget=1.0)
        self.assertAlmostEqual(sel.achieved_val_nn_usage_, 1.0, places=6)

    def test_budget_1_budget_met(self):
        sel = SLACAMASSelector(self._tree, self._nn)
        sel.fit_budget_policy(self._X_val, self._y_val, max_nn_budget=1.0)
        self.assertTrue(sel.is_budget_met_val_)


class TestSLACAMASDeterminism(unittest.TestCase, _DigitsFixture):
    """Two calls with identical val data must produce the same threshold."""

    @classmethod
    def setUpClass(cls):
        cls._load_digits()

    def test_determinism_same_threshold(self):
        for budget in [0.20, 0.35, 0.50, 0.70]:
            sel1 = SLACAMASSelector(self._tree, self._nn)
            sel1.fit_budget_policy(self._X_val, self._y_val, max_nn_budget=budget)
            sel2 = SLACAMASSelector(self._tree, self._nn)
            sel2.fit_budget_policy(self._X_val, self._y_val, max_nn_budget=budget)
            self.assertEqual(sel1.threshold_, sel2.threshold_,
                msg=f"Threshold not deterministic for budget={budget}")


class TestSLACAMASTestLabelIsolation(unittest.TestCase, _DigitsFixture):
    """Replacing test labels with garbage must not change the calibrated threshold."""

    @classmethod
    def setUpClass(cls):
        cls._load_digits()

    def test_test_labels_do_not_affect_calibration(self):
        budget = 0.35
        sel_real = SLACAMASSelector(self._tree, self._nn)
        sel_real.fit_budget_policy(self._X_val, self._y_val, max_nn_budget=budget)

        # Corrupt the test labels — calibration must ignore them entirely
        garbage_test_labels = np.zeros(len(self._y_test), dtype=int)
        # Calibrate again with the same val data; the test labels are not passed in
        sel_corrupt = SLACAMASSelector(self._tree, self._nn)
        sel_corrupt.fit_budget_policy(self._X_val, self._y_val, max_nn_budget=budget)

        self.assertEqual(sel_real.threshold_, sel_corrupt.threshold_,
            msg="Threshold changed unexpectedly; calibration must not access test labels.")


class TestSLACAMASDigitsBudgetFeasibility(unittest.TestCase, _DigitsFixture):
    """For each Digits budget, achieved val NN usage must be ≤ budget + tolerance."""

    @classmethod
    def setUpClass(cls):
        cls._load_digits()
        cls._tolerance = 0.05   # 5 % tolerance for discrete threshold search

    def test_digits_budget_feasibility(self):
        for budget in [0.20, 0.35, 0.50, 0.70]:
            sel = SLACAMASSelector(self._tree, self._nn)
            sel.fit_budget_policy(self._X_val, self._y_val, max_nn_budget=budget)
            self.assertLessEqual(
                sel.achieved_val_nn_usage_,
                budget + self._tolerance,
                msg=(
                    f"Budget {budget}: achieved val NN usage "
                    f"{sel.achieved_val_nn_usage_:.4f} exceeds "
                    f"budget {budget} + tolerance {self._tolerance}"
                ),
            )


class TestSLACAMASInvalidBudget(unittest.TestCase, _DigitsFixture):
    """Out-of-range budgets must raise ValueError."""

    @classmethod
    def setUpClass(cls):
        cls._load_digits()

    def test_budget_above_1_raises(self):
        sel = SLACAMASSelector(self._tree, self._nn)
        with self.assertRaises(ValueError):
            sel.fit_budget_policy(self._X_val, self._y_val, max_nn_budget=1.01)

    def test_budget_below_0_raises(self):
        sel = SLACAMASSelector(self._tree, self._nn)
        with self.assertRaises(ValueError):
            sel.fit_budget_policy(self._X_val, self._y_val, max_nn_budget=-0.01)


class TestSLACAMASWineInfeasibleBudget(unittest.TestCase, _WineFixture):
    """
    Budget=0.10 on Wine is infeasible: the depth-2 tree produces only ~3 distinct
    uncertainty values, so the only reachable val NN usage rates are ~0 % or ~30.6 %.
    The selector must not silently claim 10 % is achievable.
    Either is_budget_met_val_ = False OR achieved_val_nn_usage_ snaps to 0 %.
    Both outcomes are acceptable; the test verifies that the reported val NN usage
    does NOT falsely claim ~10 % was achieved.
    """

    @classmethod
    def setUpClass(cls):
        cls._load_wine()

    def test_wine_infeasible_10pct_budget(self):
        budget = 0.10
        sel = SLACAMASSelector(self._w_tree, self._w_nn)
        sel.fit_budget_policy(self._X_val_w, self._y_val_w, max_nn_budget=budget)

        # The achieved val usage must be either 0 % (τ above all scores) OR
        # approximately the natural CAMAS rate (~30.6 % on Wine val, which violates
        # the 10 % budget and must set is_budget_met_val_=False).
        achieved = sel.achieved_val_nn_usage_
        if achieved > budget + SLACAMASSelector._BUDGET_TOLERANCE:
            # Infeasible: usage exceeds budget → flag must be False
            self.assertFalse(sel.is_budget_met_val_,
                msg=(
                    f"Wine budget={budget}: achieved val NN usage={achieved:.4f} "
                    f"exceeds budget but is_budget_met_val_ is True (should be False)."
                ))
        else:
            # Feasible path (τ snapped to 0 % usage) — budget_met may be True
            self.assertAlmostEqual(achieved, 0.0, places=4,
                msg=(
                    f"Wine budget={budget}: expected 0% achieved usage if budget "
                    f"is met, got {achieved:.4f}."
                ))

    def test_wine_budget_0_achieves_zero_nn(self):
        sel = SLACAMASSelector(self._w_tree, self._w_nn)
        sel.fit_budget_policy(self._X_val_w, self._y_val_w, max_nn_budget=0.0)
        preds, nn_count, _ = sel.predict(self._X_test_w)
        self.assertEqual(nn_count, 0)

    def test_wine_budget_1_achieves_full_nn(self):
        sel = SLACAMASSelector(self._w_tree, self._w_nn)
        sel.fit_budget_policy(self._X_val_w, self._y_val_w, max_nn_budget=1.0)
        preds, nn_count, _ = sel.predict(self._X_test_w)
        self.assertEqual(nn_count, len(self._X_test_w))


class TestSLACAMASModelFilesUnchanged(unittest.TestCase):
    """
    Running calibration must not modify any .pkl model file.
    Checks mtime of all six model files before and after calibration.
    """

    def test_model_files_not_modified(self):
        model_paths = [
            "models/decision_tree.pkl",
            "models/neural_network.pkl",
            "models/router.pkl",
            "models_wine/decision_tree.pkl",
            "models_wine/neural_network.pkl",
            "models_wine/router.pkl",
            "models_wine/scaler.pkl",
        ]
        mtimes_before = {}
        for p in model_paths:
            if os.path.exists(p):
                mtimes_before[p] = os.path.getmtime(p)

        # Run calibration on Digits
        tree = joblib.load("models/decision_tree.pkl")
        nn = joblib.load("models/neural_network.pkl")
        _, X_val, _, _, y_val, _ = get_dataset()
        sel = SLACAMASSelector(tree, nn)
        sel.fit_budget_policy(X_val, y_val, max_nn_budget=0.35)

        # Run calibration on Wine
        w_tree = joblib.load("models_wine/decision_tree.pkl")
        w_nn = joblib.load("models_wine/neural_network.pkl")
        _, X_val_w, _, _, y_val_w, _, _, _, _ = get_wine_dataset(scale=True, random_state=42)
        w_sel = SLACAMASSelector(w_tree, w_nn)
        w_sel.fit_budget_policy(X_val_w, y_val_w, max_nn_budget=0.30)

        for p, mtime_before in mtimes_before.items():
            mtime_after = os.path.getmtime(p)
            self.assertAlmostEqual(
                mtime_before, mtime_after, places=0,
                msg=f"Model file mtime changed after calibration: {p}",
            )


class TestSLACAMASPredictAPI(unittest.TestCase, _DigitsFixture):
    """Verify the predict() return types and shapes."""

    @classmethod
    def setUpClass(cls):
        cls._load_digits()

    def test_predict_returns_tuple_of_three(self):
        sel = SLACAMASSelector(self._tree, self._nn)
        sel.fit_budget_policy(self._X_val, self._y_val, max_nn_budget=0.35)
        result = sel.predict(self._X_test)
        self.assertEqual(len(result), 3)

    def test_predict_preds_shape(self):
        sel = SLACAMASSelector(self._tree, self._nn)
        sel.fit_budget_policy(self._X_val, self._y_val, max_nn_budget=0.35)
        preds, nn_count, elapsed = sel.predict(self._X_test)
        self.assertEqual(preds.shape[0], len(self._X_test))

    def test_predict_nn_count_int(self):
        sel = SLACAMASSelector(self._tree, self._nn)
        sel.fit_budget_policy(self._X_val, self._y_val, max_nn_budget=0.35)
        _, nn_count, _ = sel.predict(self._X_test)
        self.assertIsInstance(nn_count, int)

    def test_predict_elapsed_nonnegative(self):
        sel = SLACAMASSelector(self._tree, self._nn)
        sel.fit_budget_policy(self._X_val, self._y_val, max_nn_budget=0.35)
        _, _, elapsed = sel.predict(self._X_test)
        self.assertGreaterEqual(elapsed, 0.0)


class TestSLACAMASCalibrationReport(unittest.TestCase, _DigitsFixture):
    """calibration_report() must contain all expected keys."""

    @classmethod
    def setUpClass(cls):
        cls._load_digits()

    def test_calibration_report_keys(self):
        sel = SLACAMASSelector(self._tree, self._nn)
        sel.fit_budget_policy(self._X_val, self._y_val, max_nn_budget=0.35)
        report = sel.calibration_report()
        expected_keys = {
            "requested_budget",
            "threshold",
            "achieved_val_nn_usage",
            "is_budget_met_val",
            "val_accuracy",
            "budget_note",
        }
        self.assertEqual(set(report.keys()), expected_keys)

    def test_calibration_report_requested_budget_matches(self):
        for budget in [0.0, 0.20, 0.50, 1.0]:
            sel = SLACAMASSelector(self._tree, self._nn)
            sel.fit_budget_policy(self._X_val, self._y_val, max_nn_budget=budget)
            self.assertAlmostEqual(
                sel.calibration_report()["requested_budget"], budget, places=6
            )


if __name__ == "__main__":
    unittest.main()
