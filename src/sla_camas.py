"""
SLA-CAMAS: Budget-Constrained Adaptive Model Selection for Green AI.

Extends the Cost-Aware Multi-Criteria Adaptive Selector (CAMAS) with an explicit
maximum neural-network invocation budget specified by the user (e.g. 20 %, 35 %).
The uncertainty threshold is selected on validation data only; test labels are never
consulted during calibration.

Research Context:
    Peykani et al., "Green Artificial Intelligence: A Comprehensive Review of Metrics,
    Tools, Challenges, Trends, and Future Prospects", Springer, 2026,
    DOI: 10.1007/s11831-026-10546-2.

Key Design Decisions
---------------------
* Budget search inverts the uncertainty distribution: for budget B, the threshold τ
  is set to the (1-B) quantile of the validation uncertainty scores.  All samples
  whose composite uncertainty score is ≥ τ are escalated to the neural network; the
  rest are served by the decision tree.

* Because uncertainty scores can take many identical values (especially with shallow
  trees that produce few distinct probability leaves), the realised validation NN usage
  may differ from the requested budget.  The achieved_val_nn_usage_ attribute and the
  is_budget_met_val_ flag record the actual outcome.

* Edge cases handled explicitly:
    - budget = 0.0  → τ set above the maximum uncertainty score (max + 1.0).
                      All samples go to the tree; nn_count = 0.
    - budget = 1.0  → τ = 0.0.
                      All samples are escalated; nn_count = len(X).
    - Infeasible budget (requested B but smallest achievable fraction is still > B):
                      is_budget_met_val_ = False, achieved_val_nn_usage_ records what
                      was actually achieved.  The selector still returns a usable
                      threshold (the one that minimises NN usage on validation).

Distribution-Shift Caveat
--------------------------
A threshold calibrated on the validation split cannot guarantee an identical NN usage
rate on the held-out test set.  Distribution shift between the two splits means the
actual test-set NN fraction may differ from the validation rate at which the threshold
was selected.  SLA-CAMAS reports both is_budget_met_val_ (validation) and allows
callers to compute is_budget_met_test_ separately from the observed test-set NN fraction.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score

# Import the shared uncertainty metric function so behaviour is identical
# to the parent CAMAS class and we never re-implement floating-point logic.
try:
    from adaptive_selector import compute_uncertainty_metrics
except ImportError:
    from src.adaptive_selector import compute_uncertainty_metrics


class SLACAMASSelector:
    """
    Budget-constrained adaptive selector using validation Pareto frontier.

    Parameters
    ----------
    tree : sklearn estimator
        Pretrained lightweight classifier that supports predict_proba.
    neural_net : sklearn estimator
        Pretrained high-capacity classifier.

    Attributes (set after fit_budget_policy)
    -----------------------------------------
    threshold_ : float
        Selected uncertainty threshold for the requested budget.
    requested_budget_ : float
        The max NN usage fraction requested by the caller.
    achieved_val_nn_usage_ : float
        Actual NN usage fraction on the validation set at threshold_.
    is_budget_met_val_ : bool
        True iff achieved_val_nn_usage_ <= requested_budget_ + tolerance.
    val_accuracy_ : float
        Accuracy on the validation set at threshold_.
    budget_note_ : str
        Human-readable note describing the budget outcome.
    """

    # Small tolerance for floating-point rounding comparisons
    _BUDGET_TOLERANCE: float = 1e-6

    def __init__(self, tree: Any, neural_net: Any) -> None:
        self.tree = tree
        self.neural_net = neural_net

        # Populated by fit_budget_policy or fit_multi_budget
        self.threshold_: float = 0.50
        self.requested_budget_: float = float("nan")
        self.achieved_val_nn_usage_: float = float("nan")
        self.is_budget_met_val_: bool = False
        self.val_accuracy_: float = float("nan")
        self.budget_note_: str = "Not yet calibrated."

    # ------------------------------------------------------------------
    # Public calibration API
    # ------------------------------------------------------------------

    def fit_budget_policy(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        max_nn_budget: float,
    ) -> "SLACAMASSelector":
        """
        Select threshold τ* on validation data such that val_nn_usage ≤ max_nn_budget.

        Parameters
        ----------
        X_val : np.ndarray of shape (n_val, n_features)
            Validation features.  MUST NOT contain test samples.
        y_val : np.ndarray of shape (n_val,)
            Validation labels.  MUST NOT contain test labels.
        max_nn_budget : float in [0.0, 1.0]
            Maximum fraction of validation samples that may be escalated to the NN.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If max_nn_budget is outside [0.0, 1.0].
        """
        if not (0.0 <= max_nn_budget <= 1.0):
            raise ValueError(
                f"max_nn_budget must be in [0.0, 1.0]; got {max_nn_budget!r}"
            )

        self.requested_budget_ = float(max_nn_budget)

        # -- Compute uncertainty scores on validation data only --------
        tree_probs = self.tree.predict_proba(X_val)
        tree_preds = self.tree.classes_[np.argmax(tree_probs, axis=1)]
        nn_preds = self.neural_net.predict(X_val)

        metrics = compute_uncertainty_metrics(tree_probs)
        uncertainty_scores = metrics["composite_uncertainty"]

        n_val = len(y_val)

        # -- Handle edge cases explicitly -------------------------------
        if max_nn_budget == 0.0:
            # Force all samples to tree: set τ above every possible score.
            tau = float(uncertainty_scores.max()) + 1.0
            achieved_usage = 0.0
            preds = tree_preds.copy()
            acc = float(accuracy_score(y_val, preds))
            note = (
                "Budget = 0 %: threshold set above maximum uncertainty score. "
                "All samples served by Decision Tree; NN invocations = 0."
            )
            met = True  # 0 ≤ 0 ✓

        elif max_nn_budget == 1.0:
            # Force all samples to NN: set τ = 0.0.
            tau = 0.0
            # Every sample satisfies U(x) >= 0.0, so all escalate.
            escalate_mask = np.ones(n_val, dtype=bool)
            achieved_usage = 1.0
            preds = tree_preds.copy()
            preds[escalate_mask] = nn_preds[escalate_mask]
            acc = float(accuracy_score(y_val, preds))
            note = (
                "Budget = 100 %: threshold = 0.0. "
                "All samples escalated to Neural Network."
            )
            met = True  # 1.0 ≤ 1.0 ✓

        else:
            # General case: find the smallest τ such that the fraction of val
            # samples with U(x) >= τ does not exceed max_nn_budget.
            #
            # Strategy:
            #   Sort uncertainty scores ascending.  For budget B over n_val samples,
            #   at most floor(B * n_val) samples should be escalated.
            #   The (1-B) quantile is the smallest τ so that at most B*n_val samples
            #   score ≥ τ.  We adjust by checking actual realised usage.

            max_nn_samples = int(np.floor(max_nn_budget * n_val))

            # Sorted ascending; threshold = value at position (n_val - max_nn_samples)
            sorted_scores = np.sort(uncertainty_scores)

            if max_nn_samples == 0:
                # No escalations allowed; τ > max score.
                tau = float(sorted_scores[-1]) + 1.0
            else:
                # τ = uncertainty score at position (n_val - max_nn_samples).
                # Samples at positions [n_val - max_nn_samples : n_val] have score ≥ τ.
                split_idx = n_val - max_nn_samples
                # Add a tiny epsilon so that samples whose score equals exactly
                # sorted_scores[split_idx] are NOT escalated.  This prevents
                # over-budget behaviour when many samples share the same discrete
                # uncertainty value at the boundary (common with shallow trees that
                # produce only a few distinct probability leaves).
                tau = float(sorted_scores[split_idx]) + 1e-10

            # Realised usage (may differ if many identical values at boundary)
            escalate_mask = uncertainty_scores >= tau
            achieved_usage = float(escalate_mask.mean())

            # Compute predictions and accuracy
            preds = tree_preds.copy()
            if escalate_mask.any():
                preds[escalate_mask] = nn_preds[escalate_mask]
            acc = float(accuracy_score(y_val, preds))

            met = achieved_usage <= max_nn_budget + self._BUDGET_TOLERANCE
            if met:
                note = (
                    f"Budget {max_nn_budget*100:.1f}%: threshold τ = {tau:.6f}, "
                    f"achieved val NN usage = {achieved_usage*100:.1f}% "
                    f"(≤ budget ✓)."
                )
            else:
                note = (
                    f"Budget {max_nn_budget*100:.1f}% NOT MET on validation: "
                    f"threshold τ = {tau:.6f}, achieved val NN usage = "
                    f"{achieved_usage*100:.1f}%. "
                    f"This may occur when many uncertainty values are identical "
                    f"(e.g. a shallow tree with few distinct probability leaves). "
                    f"The selector returns the threshold that minimises NN usage "
                    f"on validation; the budget constraint is infeasible given "
                    f"the discrete uncertainty distribution."
                )

        self.threshold_ = float(tau)
        self.achieved_val_nn_usage_ = float(achieved_usage)
        self.val_accuracy_ = float(acc)
        self.is_budget_met_val_ = bool(met)
        self.budget_note_ = note
        return self

    def fit_multi_budget(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        budgets: List[float],
    ) -> Dict[float, "SLACAMASSelector"]:
        """
        Calibrate separate policies for each requested budget.

        Parameters
        ----------
        X_val, y_val : validation data (never test data).
        budgets : list of floats in [0.0, 1.0].

        Returns
        -------
        dict mapping each budget value → a fresh SLACAMASSelector already fitted
        to that budget.  self is fitted to the last budget in the list (for
        convenience when a single budget is used via fit_budget_policy).
        """
        results: Dict[float, "SLACAMASSelector"] = {}
        for b in budgets:
            selector = SLACAMASSelector(self.tree, self.neural_net)
            selector.fit_budget_policy(X_val, y_val, max_nn_budget=b)
            results[float(b)] = selector
        # Update self to the last budget for a consistent .threshold_ attribute
        last = results[float(budgets[-1])]
        self.threshold_ = last.threshold_
        self.requested_budget_ = last.requested_budget_
        self.achieved_val_nn_usage_ = last.achieved_val_nn_usage_
        self.is_budget_met_val_ = last.is_budget_met_val_
        self.val_accuracy_ = last.val_accuracy_
        self.budget_note_ = last.budget_note_
        return results

    # ------------------------------------------------------------------
    # Inference API (mirrors CostAwareAdaptiveSelector.predict)
    # ------------------------------------------------------------------

    def predict(
        self,
        X: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Adaptive inference using the calibrated budget threshold.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        threshold : float, optional
            Override threshold (defaults to self.threshold_).

        Returns
        -------
        predictions : np.ndarray
        nn_count : int
        elapsed_seconds : float
        """
        t0 = time.perf_counter()
        tau = self.threshold_ if threshold is None else float(threshold)

        tree_probs = self.tree.predict_proba(X)
        predictions = self.tree.classes_[np.argmax(tree_probs, axis=1)].copy()

        metrics = compute_uncertainty_metrics(tree_probs)
        escalate_indices = np.flatnonzero(metrics["composite_uncertainty"] >= tau)

        nn_count = len(escalate_indices)
        if nn_count > 0:
            predictions[escalate_indices] = self.neural_net.predict(X[escalate_indices])

        elapsed = time.perf_counter() - t0
        return predictions, nn_count, elapsed

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    def calibration_report(self) -> Dict[str, Any]:
        """Returns a summary dict suitable for logging and CSV export."""
        return {
            "requested_budget": self.requested_budget_,
            "threshold": self.threshold_,
            "achieved_val_nn_usage": self.achieved_val_nn_usage_,
            "is_budget_met_val": self.is_budget_met_val_,
            "val_accuracy": self.val_accuracy_,
            "budget_note": self.budget_note_,
        }
