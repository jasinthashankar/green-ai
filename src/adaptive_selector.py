"""
Cost-Aware Multi-Criteria Adaptive Selection (CAMAS) for Green AI.

Motivated by the multi-tier evaluation principles and dynamic compute recommendations in:
'Green Artificial Intelligence: A Comprehensive Review of Metrics, Tools, Challenges,
Trends, and Future Prospects' (2026), Springer, DOI: 10.1007/s11831-026-10546-2.

This module provides a calibrated, cost-utility-driven model selection framework
that evaluates prediction entropy, confidence margin, and top-1 confidence to
dynamically route inference between a cheap Decision Tree and an expensive Neural Network.
"""

import time
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.metrics import accuracy_score


def compute_uncertainty_metrics(probabilities: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Computes fine-grained uncertainty metrics for class probability distributions.

    Parameters
    ----------
    probabilities : np.ndarray of shape (n_samples, n_classes)
        Output probabilities from the lightweight classifier.

    Returns
    -------
    dict containing:
        - 'confidence': Top-1 predicted probability.
        - 'margin': Difference between top-1 and top-2 predicted probabilities.
        - 'normalized_entropy': Shannon entropy normalized to [0, 1] by ln(n_classes).
        - 'composite_uncertainty': Weighted combination of uncertainty signals.
    """
    probs = np.clip(probabilities, 1e-12, 1.0)
    n_classes = probs.shape[1]

    # Sort probabilities along class axis in ascending order
    sorted_probs = np.sort(probs, axis=1)
    p_top1 = sorted_probs[:, -1]
    p_top2 = sorted_probs[:, -2] if n_classes > 1 else np.zeros_like(p_top1)

    # Margin: gap between best and second-best candidate
    margin = p_top1 - p_top2

    # Normalized Shannon Entropy: 0 when completely certain, 1 when uniform
    raw_entropy = -np.sum(probs * np.log(probs), axis=1)
    max_entropy = np.log(n_classes) if n_classes > 1 else 1.0
    normalized_entropy = raw_entropy / max_entropy

    # Normalized confidence deficiency in [0, 1]
    # For K classes, p_top1 min is 1/K (uniform) and max is 1.0 (certain)
    min_top1 = 1.0 / n_classes if n_classes > 1 else 0.0
    conf_range = 1.0 - min_top1
    conf_deficiency = (1.0 - p_top1) / conf_range if conf_range > 0 else (1.0 - p_top1)

    # Composite uncertainty score U in [0, 1]
    # High score = high ambiguity / high need for high-capacity model
    # w1 * conf_deficiency + w2 * normalized_entropy + w3 * (1 - margin)
    # Balanced default weights: 0.35, 0.35, 0.30
    composite_uncertainty = (
        0.35 * conf_deficiency +
        0.35 * normalized_entropy +
        0.30 * (1.0 - margin)
    )

    return {
        "confidence": p_top1,
        "margin": margin,
        "normalized_entropy": normalized_entropy,
        "composite_uncertainty": composite_uncertainty,
    }


class CostAwareAdaptiveSelector:
    """
    Cost-Aware Multi-Criteria Adaptive Model Selector (CAMAS).

    Balances classification accuracy against computational inference budget
    via a tunable Green Penalty parameter lambda in [0, 1].

    Parameters
    ----------
    tree : object
        Pretrained lightweight classifier (e.g., DecisionTreeClassifier).
    neural_net : object
        Pretrained high-capacity classifier (e.g., MLPClassifier).
    cost_ratio : float
        Normalized relative computational cost of neural_net vs tree (default 1.0).
    """

    def __init__(self, tree, neural_net, cost_ratio: float = 1.0):
        self.tree = tree
        self.neural_net = neural_net
        self.cost_ratio = cost_ratio
        self.threshold_: float = 0.50
        self.lambda_penalty_: float = 0.50
        self.pareto_frontier_: List[Dict[str, float]] = []

    def fit_policy(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        lambda_penalty: float = 0.35,
        grid_points: int = 100,
    ) -> "CostAwareAdaptiveSelector":
        """
        Calibrates the optimal uncertainty decision threshold on validation data.

        Maximizes the Cost-Utility Objective:
            Utility(tau) = Accuracy(tau) - lambda_penalty * NN_Usage(tau)

        Strict Data Hygiene:
            Calibration is conducted strictly on validation data to prevent
            data leakage to the held-out test benchmark.
        """
        self.lambda_penalty_ = lambda_penalty

        tree_probs = self.tree.predict_proba(X_val)
        tree_preds = self.tree.classes_[np.argmax(tree_probs, axis=1)]
        nn_preds = self.neural_net.predict(X_val)

        metrics = compute_uncertainty_metrics(tree_probs)
        uncertainty_scores = metrics["composite_uncertainty"]

        # Search across threshold candidates
        candidate_thresholds = np.linspace(
            uncertainty_scores.min(),
            uncertainty_scores.max() + 1e-4,
            grid_points
        )

        best_utility = -float("inf")
        best_threshold = 0.50
        frontier = []

        n_samples = len(y_val)

        for tau in candidate_thresholds:
            escalate_mask = uncertainty_scores >= tau
            nn_usage = escalate_mask.mean()

            preds = tree_preds.copy()
            if escalate_mask.any():
                preds[escalate_mask] = nn_preds[escalate_mask]

            acc = accuracy_score(y_val, preds)
            # Cost-Utility formulation:
            utility = acc - (lambda_penalty * self.cost_ratio * nn_usage)

            frontier.append({
                "threshold": float(tau),
                "val_accuracy": float(acc),
                "val_nn_usage": float(nn_usage),
                "utility": float(utility),
            })

            if utility > best_utility:
                best_utility = utility
                best_threshold = tau

        self.threshold_ = float(best_threshold)
        self.pareto_frontier_ = frontier
        return self

    def predict(
        self,
        X: np.ndarray,
        threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Performs vectorized, single-pass adaptive inference.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Inference input samples.
        threshold : Optional[float]
            Optional threshold override; defaults to self.threshold_.

        Returns
        -------
        predictions : np.ndarray of shape (n_samples,)
        nn_count : int (number of samples escalated to Neural Network)
        elapsed_seconds : float
        """
        t0 = time.perf_counter()
        tau = self.threshold_ if threshold is None else threshold

        # Step 1: Execute lightweight model once
        tree_probs = self.tree.predict_proba(X)
        predictions = self.tree.classes_[np.argmax(tree_probs, axis=1)]

        # Step 2: Compute composite multi-criteria uncertainty
        metrics = compute_uncertainty_metrics(tree_probs)
        escalate_indices = np.flatnonzero(metrics["composite_uncertainty"] >= tau)

        # Step 3: Escalate only ambiguous / high-uncertainty samples to the heavy model
        nn_count = len(escalate_indices)
        if nn_count > 0:
            predictions[escalate_indices] = self.neural_net.predict(X[escalate_indices])

        elapsed = time.perf_counter() - t0
        return predictions, nn_count, elapsed

    def get_pareto_profiles(self) -> Dict[str, Dict[str, float]]:
        """
        Extracts representative operational profiles along the validation Pareto frontier:
        - 'max_accuracy': Minimal threshold (escalates readily for peak accuracy).
        - 'balanced_green': Optimal threshold balancing accuracy gain against NN cost.
        - 'ultra_green': High threshold (minimizes computation, maximizing energy savings).
        """
        if not self.pareto_frontier_:
            return {}

        sorted_by_acc = sorted(self.pareto_frontier_, key=lambda r: (-r["val_accuracy"], r["val_nn_usage"]))
        sorted_by_util = sorted(self.pareto_frontier_, key=lambda r: -r["utility"])
        sorted_by_usage = sorted(self.pareto_frontier_, key=lambda r: (r["val_nn_usage"], -r["val_accuracy"]))

        return {
            "max_accuracy": sorted_by_acc[0],
            "balanced_green": sorted_by_util[0],
            "ultra_green": sorted_by_usage[0],
        }


def get_theoretical_flops(n_samples: int, nn_usage_fraction: float) -> Dict[str, float]:
    """
    Computes theoretical FLOPs per sample based on model architectures:
    - Decision Tree (depth 5): ~5 comparisons per sample.
    - MLP (64 -> 256 -> 128 -> 64 -> 32 -> 16 -> 10):
      Linear layers MACs = 64*256 + 256*128 + 128*64 + 64*32 + 32*16 + 16*10 = 60,064 MACs.
      FLOPs ≈ 2 * MACs = 120,128 FLOPs per sample.
    """
    tree_flops_per_sample = 5.0
    nn_flops_per_sample = 120128.0

    effective_flops_per_sample = (
        tree_flops_per_sample + (nn_usage_fraction * nn_flops_per_sample)
    )

    return {
        "tree_flops": tree_flops_per_sample,
        "nn_flops": nn_flops_per_sample,
        "effective_flops_per_sample": effective_flops_per_sample,
        "total_batch_flops": effective_flops_per_sample * n_samples,
        "theoretical_flops_reduction_pct": (1.0 - effective_flops_per_sample / nn_flops_per_sample) * 100.0,
    }


if __name__ == "__main__":
    from dataset import get_dataset

    print("Loading models and validation dataset...")
    tree_model = joblib.load("models/decision_tree.pkl")
    nn_model = joblib.load("models/neural_network.pkl")
    _, X_val, X_test, _, y_val, y_test = get_dataset()

    selector = CostAwareAdaptiveSelector(tree_model, nn_model)
    # Calibrate policy on validation set
    selector.fit_policy(X_val, y_val, lambda_penalty=0.30)

    print("\n--- CAMAS Policy Calibration (Validation Set) ---")
    print(f"Optimal Threshold: {selector.threshold_:.4f}")
    profiles = selector.get_pareto_profiles()
    for profile_name, stats in profiles.items():
        print(f"  {profile_name:<16}: Val Acc = {stats['val_accuracy']*100:.2f}%, "
              f"NN Usage = {stats['val_nn_usage']*100:.1f}%, Utility = {stats['utility']:.4f}")

    # Evaluate on held-out test set
    preds, nn_used, latency = selector.predict(X_test)
    test_acc = accuracy_score(y_test, preds)
    nn_pct = nn_used / len(X_test) * 100.0
    flops = get_theoretical_flops(len(X_test), nn_used / len(X_test))

    print("\n--- CAMAS Evaluation on Held-Out Test Set ---")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Samples escalated to NN: {nn_used} / {len(X_test)} ({nn_pct:.1f}%)")
    print(f"Inference latency: {latency*1000:.3f} ms")
    print(f"Theoretical FLOPs reduction vs NN: {flops['theoretical_flops_reduction_pct']:.1f}%")
