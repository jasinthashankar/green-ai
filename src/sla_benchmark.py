"""
SLA-CAMAS Six-Method Benchmark for Green AI Dynamic Model Selection.

Compares six methods on both Digits and Wine held-out test sets:
  1. Decision Tree
  2. Neural Network
  3. Dynamic Cascade
  4. Dynamic Routing
  5. CAMAS (Adaptive Selection)
  6. SLA-CAMAS @ budget B  (one row per budget)

Research Context:
    Peykani et al., "Green Artificial Intelligence: A Comprehensive Review of Metrics,
    Tools, Challenges, Trends, and Future Prospects", Springer, 2026,
    DOI: 10.1007/s11831-026-10546-2.

Strict Constraints
------------------
* Models loaded read-only; no retraining.
* Validation data only used for CAMAS/SLA-CAMAS threshold calibration.
* Test labels never consulted during calibration.
* New outputs: results/sla_camas_digits.csv, results/sla_camas_wine.csv only.
* Existing CSV files are never modified or deleted.

Distribution-Shift Caveat (documented inline and in output)
------------------------------------------------------------
A threshold selected on the validation split cannot guarantee the same NN usage
rate on the test split.  Both val and test NN usage fractions are reported for
each SLA-CAMAS budget so readers can observe any discrepancy.
"""

import os
import sys
import time
import statistics
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# --------------------------------------------------------------------------
# Path setup: support both `python src/sla_benchmark.py` and module imports
# --------------------------------------------------------------------------
_HERE = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
for _p in [_HERE, os.path.join(_ROOT, "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from dataset import get_dataset
    from dataset_wine import get_wine_dataset
    from adaptive_selector import CostAwareAdaptiveSelector, compute_uncertainty_metrics
    from sla_camas import SLACAMASSelector
    from wine_experiment import (
        load_wine_models,
        build_wine_runners,
        compute_wine_theoretical_complexity,
    )
    from enhanced_benchmark import (
        load_pretrained_assets,
        build_method_runners,
        compute_theoretical_complexity,
        measure_latency_distribution,
    )
except ImportError:
    from src.dataset import get_dataset
    from src.dataset_wine import get_wine_dataset
    from src.adaptive_selector import CostAwareAdaptiveSelector, compute_uncertainty_metrics
    from src.sla_camas import SLACAMASSelector
    from src.wine_experiment import (
        load_wine_models,
        build_wine_runners,
        compute_wine_theoretical_complexity,
    )
    from src.enhanced_benchmark import (
        load_pretrained_assets,
        build_method_runners,
        compute_theoretical_complexity,
        measure_latency_distribution,
    )


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

DIGITS_BUDGETS = [0.20, 0.35, 0.50, 0.70]
WINE_BUDGETS   = [0.00, 0.30, 0.50, 1.00]

# Validated against enhanced_benchmark.py / wine_experiment.py
DIGITS_CAMAS_LAMBDA = 0.20
WINE_CAMAS_LAMBDA   = 0.08
CASCADE_THRESHOLD   = 0.80

DISTRIBUTION_SHIFT_CAVEAT = (
    "IMPORTANT: The SLA-CAMAS uncertainty threshold is calibrated on the validation split. "
    "Due to potential distribution shift between validation and test splits, the realised "
    "test-set NN usage fraction may differ from the validation-set NN usage fraction at which "
    "the threshold was selected.  Both fractions are reported for transparency.  "
    "No exact test-set budget guarantee is implied."
)

WINE_DISCRETE_UNCERTAINTY_NOTE = (
    "NOTE (Wine): The depth-2 Decision Tree produces only ~3 distinct probability leaves, "
    "resulting in ~3 unique composite uncertainty values.  Budget requests between 0 % and "
    "~30.6 % (the natural NN usage rate of CAMAS) are not continuously achievable; the "
    "realised NN usage will snap to the nearest feasible discrete value (0 % or ~30.6 %).  "
    "This limitation is documented per the 2026 Springer Green AI review's recommendation "
    "to report measurement uncertainty and system constraints explicitly."
)


# --------------------------------------------------------------------------
# Helper: build SLA-CAMAS runner for a single calibrated selector
# --------------------------------------------------------------------------

def _make_sla_runner(
    selector: SLACAMASSelector,
) -> Callable[[np.ndarray], Tuple[np.ndarray, int]]:
    """Returns a (preds, nn_count) runner for a calibrated SLACAMASSelector."""
    def _run(X: np.ndarray) -> Tuple[np.ndarray, int]:
        preds, nn_count, _ = selector.predict(X)
        return preds, int(nn_count)
    return _run


# --------------------------------------------------------------------------
# Digits benchmark
# --------------------------------------------------------------------------

def run_digits_sla_benchmark(
    repeats: int = 150,
    warmup: int = 20,
    output_dir: str = "results",
    output_filename: str = "sla_camas_digits.csv",
) -> pd.DataFrame:
    """
    Six-method benchmark on the Digits held-out test set (360 samples).

    Returns a DataFrame with one row per method (and one row per SLA budget).
    Saves to results/sla_camas_digits.csv (never overwrites existing files).
    """
    # ---- Load models (read-only) ----
    tree, neural_net, router, _, X_val, X_test, _, y_val, y_test = (
        load_pretrained_assets(models_dir="models")
    )

    # ---- Calibrate CAMAS (baseline) on validation data ----
    camas = CostAwareAdaptiveSelector(tree, neural_net)
    camas.fit_policy(X_val, y_val, lambda_penalty=DIGITS_CAMAS_LAMBDA)

    # ---- Build standard 5-method runners ----
    base_runners = build_method_runners(
        tree, neural_net, router, camas, cascade_threshold=CASCADE_THRESHOLD
    )

    # ---- Calibrate SLA-CAMAS for each budget on validation data ----
    sla_selectors: Dict[float, SLACAMASSelector] = {}
    for budget in DIGITS_BUDGETS:
        sel = SLACAMASSelector(tree, neural_net)
        sel.fit_budget_policy(X_val, y_val, max_nn_budget=budget)
        sla_selectors[budget] = sel

    # ---- Combine all runners for latency measurement ----
    all_runners: Dict[str, Callable] = dict(base_runners)
    for budget, sel in sla_selectors.items():
        label = f"SLA-CAMAS @ {int(budget*100)}%"
        all_runners[label] = _make_sla_runner(sel)

    # ---- Evaluate accuracy & NN usage on test set ----
    n_test = len(y_test)
    pred_data: Dict[str, Dict] = {}

    for name, fn in all_runners.items():
        preds, nn_count = fn(X_test)
        acc = float(accuracy_score(y_test, preds) * 100.0)
        nn_pct = (nn_count / n_test) * 100.0
        complexity = compute_theoretical_complexity(name, nn_count, n_test)
        pred_data[name] = {
            "accuracy": acc,
            "nn_count": nn_count,
            "nn_usage_pct": nn_pct,
            "complexity": complexity,
        }

    # ---- Latency distribution (warm-up + repeated trials) ----
    latency_stats = measure_latency_distribution(
        all_runners, X_test, repeats=repeats, warmup=warmup
    )

    # ---- Assemble rows ----
    nn_latency_median = latency_stats["Neural Network"]["median"]
    nn_accuracy       = pred_data["Neural Network"]["accuracy"]
    rows = []

    # 5 baseline methods
    for name in base_runners:
        row = _build_row(
            name=name,
            pred_data=pred_data[name],
            lat=latency_stats[name],
            n_test=n_test,
            nn_latency_median=nn_latency_median,
            nn_accuracy=nn_accuracy,
            dataset="Digits",
            sla_budget=None,
            val_nn_usage=None,
            is_budget_met_val=None,
            budget_note=None,
            threshold=camas.threshold_ if name == "CAMAS (Adaptive Selection)" else None,
        )
        rows.append(row)

    # SLA-CAMAS rows
    for budget, sel in sla_selectors.items():
        label = f"SLA-CAMAS @ {int(budget*100)}%"
        test_nn_pct = pred_data[label]["nn_usage_pct"] / 100.0
        is_budget_met_test = test_nn_pct <= budget + SLACAMASSelector._BUDGET_TOLERANCE
        row = _build_row(
            name=label,
            pred_data=pred_data[label],
            lat=latency_stats[label],
            n_test=n_test,
            nn_latency_median=nn_latency_median,
            nn_accuracy=nn_accuracy,
            dataset="Digits",
            sla_budget=budget,
            val_nn_usage=sel.achieved_val_nn_usage_,
            is_budget_met_val=sel.is_budget_met_val_,
            is_budget_met_test=is_budget_met_test,
            budget_note=sel.budget_note_,
            threshold=sel.threshold_,
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    _save_csv(df, output_dir, output_filename)
    return df


# --------------------------------------------------------------------------
# Wine benchmark
# --------------------------------------------------------------------------

def run_wine_sla_benchmark(
    repeats: int = 150,
    warmup: int = 20,
    output_dir: str = "results",
    output_filename: str = "sla_camas_wine.csv",
) -> pd.DataFrame:
    """
    Six-method benchmark on the Wine held-out test set (36 samples).

    Returns a DataFrame with one row per method (and one row per SLA budget).
    Saves to results/sla_camas_wine.csv (never overwrites existing files).
    """
    # ---- Load Wine models (read-only) ----
    tree, neural_net, router, _ = load_wine_models(models_dir="models_wine")
    _, X_val, X_test, _, y_val, y_test, _, _, _ = get_wine_dataset(
        scale=True, random_state=42
    )

    # ---- Calibrate CAMAS (baseline) on validation data ----
    camas = CostAwareAdaptiveSelector(tree, neural_net)
    camas.fit_policy(X_val, y_val, lambda_penalty=WINE_CAMAS_LAMBDA)

    # ---- Build standard 5-method runners ----
    base_runners = build_wine_runners(
        tree, neural_net, router, camas, cascade_threshold=CASCADE_THRESHOLD
    )

    # ---- Calibrate SLA-CAMAS for each budget on validation data ----
    sla_selectors: Dict[float, SLACAMASSelector] = {}
    for budget in WINE_BUDGETS:
        sel = SLACAMASSelector(tree, neural_net)
        sel.fit_budget_policy(X_val, y_val, max_nn_budget=budget)
        sla_selectors[budget] = sel

    # ---- Combine all runners ----
    all_runners: Dict[str, Callable] = dict(base_runners)
    for budget, sel in sla_selectors.items():
        label = f"SLA-CAMAS @ {int(budget*100)}%"
        all_runners[label] = _make_sla_runner(sel)

    # ---- Evaluate accuracy & NN usage on test set ----
    n_test = len(y_test)
    pred_data: Dict[str, Dict] = {}

    for name, fn in all_runners.items():
        preds, nn_count = fn(X_test)
        acc = float(accuracy_score(y_test, preds) * 100.0)
        nn_pct = (nn_count / n_test) * 100.0
        complexity = compute_wine_theoretical_complexity(name, nn_count, n_test)
        pred_data[name] = {
            "accuracy": acc,
            "nn_count": nn_count,
            "nn_usage_pct": nn_pct,
            "complexity": complexity,
        }

    # ---- Latency distribution ----
    latency_stats = measure_latency_distribution(
        all_runners, X_test, repeats=repeats, warmup=warmup
    )

    # ---- Assemble rows ----
    nn_latency_median = latency_stats["Neural Network"]["median"]
    nn_accuracy       = pred_data["Neural Network"]["accuracy"]
    rows = []

    for name in base_runners:
        row = _build_row(
            name=name,
            pred_data=pred_data[name],
            lat=latency_stats[name],
            n_test=n_test,
            nn_latency_median=nn_latency_median,
            nn_accuracy=nn_accuracy,
            dataset="Wine",
            sla_budget=None,
            val_nn_usage=None,
            is_budget_met_val=None,
            budget_note=None,
            threshold=camas.threshold_ if name == "CAMAS (Adaptive Selection)" else None,
        )
        rows.append(row)

    for budget, sel in sla_selectors.items():
        label = f"SLA-CAMAS @ {int(budget*100)}%"
        test_nn_pct = pred_data[label]["nn_usage_pct"] / 100.0
        is_budget_met_test = test_nn_pct <= budget + SLACAMASSelector._BUDGET_TOLERANCE
        row = _build_row(
            name=label,
            pred_data=pred_data[label],
            lat=latency_stats[label],
            n_test=n_test,
            nn_latency_median=nn_latency_median,
            nn_accuracy=nn_accuracy,
            dataset="Wine",
            sla_budget=budget,
            val_nn_usage=sel.achieved_val_nn_usage_,
            is_budget_met_val=sel.is_budget_met_val_,
            is_budget_met_test=is_budget_met_test,
            budget_note=sel.budget_note_,
            threshold=sel.threshold_,
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    _save_csv(df, output_dir, output_filename)
    return df


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def _build_row(
    name: str,
    pred_data: Dict,
    lat: Dict[str, float],
    n_test: int,
    nn_latency_median: float,
    nn_accuracy: float,
    dataset: str,
    sla_budget: Optional[float],
    val_nn_usage: Optional[float],
    is_budget_met_val: Optional[bool],
    budget_note: Optional[str],
    threshold: Optional[float],
    is_budget_met_test: Optional[bool] = None,
) -> Dict:
    time_saved_vs_nn = (1.0 - (lat["median"] / nn_latency_median)) * 100.0
    acc_kept_vs_nn   = (pred_data["accuracy"] / nn_accuracy) * 100.0

    return {
        "Dataset":                    dataset,
        "Method":                     name,
        "SLA Budget (%)":             round(sla_budget * 100, 1) if sla_budget is not None else "N/A",
        "Threshold (tau)":            round(threshold, 6) if threshold is not None else "N/A",
        "Val NN Usage (%)":           round(val_nn_usage * 100, 1) if val_nn_usage is not None else "N/A",
        "Budget Met (Val)":           str(is_budget_met_val) if is_budget_met_val is not None else "N/A",
        "Accuracy (%)":               round(pred_data["accuracy"], 2),
        "Test Samples":               n_test,
        "NN Invocations":             pred_data["nn_count"],
        "NN Usage (%) [Test]":        round(pred_data["nn_usage_pct"], 1),
        "Budget Met (Test)":          str(is_budget_met_test) if is_budget_met_test is not None else "N/A",
        "Latency Mean (ms)":          round(lat["mean"], 4),
        "Latency Std (ms)":           round(lat["std"], 4),
        "Latency Median (ms)":        round(lat["median"], 4),
        "Latency Min (ms)":           round(lat["min"], 4),
        "Latency Max (ms)":           round(lat["max"], 4),
        "Latency Saved vs NN (%)":    round(time_saved_vs_nn, 1),
        "Accuracy Kept vs NN (%)":    round(acc_kept_vs_nn, 1),
        "Theoretical FLOPs/Sample":   int(pred_data["complexity"]["flops_per_sample"]),
        "Theoretical FLOPs Saved (%)": round(pred_data["complexity"]["flops_reduction_pct"], 1),
        "Budget Note":                budget_note if budget_note is not None else "",
    }


def _save_csv(df: pd.DataFrame, output_dir: str, filename: str) -> str:
    """Save to results/ without overwriting existing files."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    if os.path.exists(out_path):
        raise FileExistsError(
            f"Output file already exists: {out_path}\n"
            "SLA benchmark refuses to overwrite existing results. "
            "Delete or rename the file to proceed."
        )
    df.to_csv(out_path, index=False)
    return out_path


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 65)
    print("  SLA-CAMAS SIX-METHOD BENCHMARK (PHASE 3)")
    print("=" * 65)
    print(f"Digits budgets : {[f'{b*100:.0f}%' for b in DIGITS_BUDGETS]}")
    print(f"Wine budgets   : {[f'{b*100:.0f}%' for b in WINE_BUDGETS]}")
    print()
    print(DISTRIBUTION_SHIFT_CAVEAT)
    print()
    print(WINE_DISCRETE_UNCERTAINTY_NOTE)
    print()

    # ---- Digits ----
    print("-" * 65)
    print("Running DIGITS benchmark ...")
    digits_df = run_digits_sla_benchmark(
        repeats=150, warmup=20,
        output_dir="results",
        output_filename="sla_camas_digits.csv",
    )
    print("\n--- DIGITS RESULTS ---")
    _print_summary(digits_df)
    print(f"\nSaved: results/sla_camas_digits.csv")

    # ---- Wine ----
    print("\n" + "-" * 65)
    print("Running WINE benchmark ...")
    wine_df = run_wine_sla_benchmark(
        repeats=150, warmup=20,
        output_dir="results",
        output_filename="sla_camas_wine.csv",
    )
    print("\n--- WINE RESULTS ---")
    _print_summary(wine_df)
    print(f"\nSaved: results/sla_camas_wine.csv")

    print("\n" + "=" * 65)
    print("Phase 3 complete.  Both CSV files written to results/.")
    print("=" * 65)


def _print_summary(df: pd.DataFrame) -> None:
    cols = [
        "Method", "SLA Budget (%)", "Threshold (tau)",
        "Val NN Usage (%)", "Budget Met (Val)",
        "Accuracy (%)", "NN Usage (%) [Test]", "Budget Met (Test)",
        "Latency Median (ms)", "Theoretical FLOPs Saved (%)",
    ]
    available = [c for c in cols if c in df.columns]
    print(df[available].to_string(index=False))


if __name__ == "__main__":
    main()
