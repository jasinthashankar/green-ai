"""
Wine Generalization Experiment for Green AI Dynamic Model Selection.

Evaluates whether the adaptive model-selection paradigm (and specifically CAMAS)
generalizes to a secondary domain (chemical cultivar classification: Wine).

Research Context:
Peykani et al., 'Green Artificial Intelligence: A Comprehensive Review of Metrics,
Tools, Challenges, Trends, and Future Prospects', Springer, 2026,
DOI: 10.1007/s11831-026-10546-2.

Generalization Scope & Limitations Notice:
- Wine dataset contains 178 samples, 13 features, and 3 classes.
- A small tabular dataset demonstrates cross-domain feasibility and algorithmic transferability
  (preliminary evidence), but does NOT constitute proof of broad generality across high-dimensional
  modalities (vision, NLP, large audio).
- Measurements with CodeCarbon on microsecond workloads reflect software TDP/heuristic estimates,
  not physical micro-joule sensor readings.

Strict Isolation:
- Models saved exclusively to models_wine/ (never models/).
- Results saved exclusively to results/wine_benchmark*.csv (never existing digits files).
- Preprocessing (StandardScaler) fitted strictly on training data.
- Policy tuning & routing trained strictly on validation data.
"""

import argparse
import os
import sys
import time
import statistics
from typing import Any, Callable, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from dataset_wine import get_wine_dataset
from adaptive_selector import CostAwareAdaptiveSelector

try:
    from codecarbon import OfflineEmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False


WINE_LIMITATIONS = {
    "dataset_scale": "Wine is small (178 samples total, 36 test samples); serves as preliminary cross-domain validation, not universal proof.",
    "software_tracker": "CodeCarbon OfflineEmissionsTracker (country ISO: IND)",
    "sensor_status": "Windows software heuristic / constant TDP model; scales proportionally with execution runtime.",
    "micro_inference": "Sub-millisecond inference time requires aggregated repeat loops for software emission tracking.",
    "flops_vs_energy": "Theoretical FLOPs represent algorithmic reduction; hardware cache and interpreter overhead affect empirical time.",
}


def train_wine_models(
    models_dir: str = "models_wine",
    random_state: int = 42,
) -> Tuple[DecisionTreeClassifier, MLPClassifier, LogisticRegression, Any]:
    """
    Trains separate Decision Tree and MLP Neural Network for Wine data.
    Trains Logistic Regression router strictly on validation data.
    Saves models exclusively in models_wine/ directory.
    """
    os.makedirs(models_dir, exist_ok=True)

    X_train, X_val, X_test, y_train, y_val, y_test, scaler, _, _ = get_wine_dataset(
        scale=True, random_state=random_state
    )

    # 1. Lightweight Model: Decision Tree (depth 2 for small tabular dataset)
    tree = DecisionTreeClassifier(max_depth=2, random_state=random_state)
    tree.fit(X_train, y_train)

    # 2. High-Capacity Model: MLP Classifier (64, 32 hidden units)
    neural_net = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=1000,
        random_state=random_state,
    )
    neural_net.fit(X_train, y_train)

    # 3. Router: Logistic Regression trained strictly on validation split
    tree_val_probs = tree.predict_proba(X_val)
    tree_val_preds = tree.predict(X_val)
    nn_val_preds = neural_net.predict(X_val)

    # Target: 1 if Tree failed and NN succeeded; else 0
    router_labels = ((tree_val_preds != y_val) & (nn_val_preds == y_val)).astype(int)

    # In case validation has only one class, handle gracefully
    if len(np.unique(router_labels)) > 1:
        router = LogisticRegression(class_weight="balanced", random_state=random_state)
        router.fit(tree_val_probs, router_labels)
    else:
        # Fallback trivial selector
        router = LogisticRegression(random_state=random_state)
        router.fit(tree_val_probs, np.zeros(len(y_val), dtype=int))

    # Save to models_wine/
    joblib.dump(tree, os.path.join(models_dir, "decision_tree.pkl"))
    joblib.dump(neural_net, os.path.join(models_dir, "neural_network.pkl"))
    joblib.dump(router, os.path.join(models_dir, "router.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))

    return tree, neural_net, router, scaler


def load_wine_models(
    models_dir: str = "models_wine",
) -> Tuple[DecisionTreeClassifier, MLPClassifier, LogisticRegression, Any]:
    """
    Loads saved Wine models without retraining.
    """
    tree_path = os.path.join(models_dir, "decision_tree.pkl")
    nn_path = os.path.join(models_dir, "neural_network.pkl")
    router_path = os.path.join(models_dir, "router.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")

    if not all(os.path.exists(p) for p in [tree_path, nn_path, router_path, scaler_path]):
        raise FileNotFoundError(f"One or more Wine models missing in {models_dir}")

    tree = joblib.load(tree_path)
    neural_net = joblib.load(nn_path)
    router = joblib.load(router_path)
    scaler = joblib.load(scaler_path)
    return tree, neural_net, router, scaler


def compute_wine_theoretical_complexity(
    method_name: str,
    nn_count: int,
    total_samples: int = 36
) -> Dict[str, float]:
    """
    Computes theoretical FLOPs for the Wine dataset architecture:
    - Input features: 13
    - Classes: 3
    - Decision Tree (depth 2): at most 2 threshold comparisons per sample.
    - Neural Network (13 -> 64 -> 32 -> 3):
      Layer 1: 13 * 64 = 832 MACs
      Layer 2: 64 * 32 = 2,048 MACs
      Layer 3: 32 * 3 = 96 MACs
      Total MACs = 2,976. Total FLOPs per sample = 2 * 2,976 = 5,952 FLOPs.
    - Dynamic Cascade: Tree comparisons (2) + NN FLOPs on escalated samples.
    - Dynamic Routing: Tree (2) + Router (3 classes: 6 FLOPs) + NN on routed samples.
    - CAMAS: Tree (2) + Uncertainty calculation (15 FLOPs) + NN on escalated samples.
    """
    tree_flops_per_sample = 2.0
    nn_flops_per_sample = 5952.0

    if method_name == "Decision Tree":
        total_flops = tree_flops_per_sample * total_samples
    elif method_name == "Neural Network":
        total_flops = nn_flops_per_sample * total_samples
    elif method_name == "Dynamic Cascade":
        total_flops = (tree_flops_per_sample * total_samples) + (nn_count * nn_flops_per_sample)
    elif method_name == "Dynamic Routing":
        router_overhead_flops = 6.0 * total_samples
        total_flops = (tree_flops_per_sample * total_samples) + router_overhead_flops + (nn_count * nn_flops_per_sample)
    elif "CAMAS" in method_name:
        camas_overhead_flops = 15.0 * total_samples
        total_flops = (tree_flops_per_sample * total_samples) + camas_overhead_flops + (nn_count * nn_flops_per_sample)
    else:
        total_flops = nn_flops_per_sample * total_samples

    flops_per_sample = total_flops / total_samples
    baseline_nn_flops = nn_flops_per_sample * total_samples
    flops_reduction_pct = (1.0 - (total_flops / baseline_nn_flops)) * 100.0

    return {
        "total_batch_flops": total_flops,
        "flops_per_sample": flops_per_sample,
        "flops_reduction_pct": flops_reduction_pct,
    }


def build_wine_runners(
    tree: Any,
    neural_net: Any,
    router: Any,
    camas_selector: CostAwareAdaptiveSelector,
    cascade_threshold: float = 0.80,
) -> Dict[str, Callable[[np.ndarray], Tuple[np.ndarray, int]]]:
    """
    Builds standardized runner functions for all five methods on Wine data.
    """
    def run_tree(X: np.ndarray) -> Tuple[np.ndarray, int]:
        return tree.predict(X), 0

    def run_nn(X: np.ndarray) -> Tuple[np.ndarray, int]:
        return neural_net.predict(X), len(X)

    def run_cascade(X: np.ndarray) -> Tuple[np.ndarray, int]:
        probs = tree.predict_proba(X)
        preds = tree.classes_[probs.argmax(axis=1)]
        hard_mask = probs.max(axis=1) < cascade_threshold
        hard_count = int(hard_mask.sum())
        if hard_count > 0:
            preds[hard_mask] = neural_net.predict(X[hard_mask])
        return preds, hard_count

    def run_routing(X: np.ndarray) -> Tuple[np.ndarray, int]:
        probs = tree.predict_proba(X)
        preds = tree.classes_[probs.argmax(axis=1)]
        use_nn_mask = (router.predict(probs) == 1)
        nn_count = int(use_nn_mask.sum())
        if nn_count > 0:
            preds[use_nn_mask] = neural_net.predict(X[use_nn_mask])
        return preds, nn_count

    def run_camas(X: np.ndarray) -> Tuple[np.ndarray, int]:
        preds, nn_count, _ = camas_selector.predict(X)
        return preds, int(nn_count)

    return {
        "Decision Tree": run_tree,
        "Neural Network": run_nn,
        "Dynamic Cascade": run_cascade,
        "Dynamic Routing": run_routing,
        "CAMAS (Adaptive Selection)": run_camas,
    }


def measure_wine_latency(
    runners: Dict[str, Callable[[np.ndarray], Tuple[np.ndarray, int]]],
    X: np.ndarray,
    repeats: int = 150,
    warmup: int = 20,
) -> Dict[str, Dict[str, float]]:
    """
    Measures latency on Wine test batch with warm-up and interleaved order.
    """
    for fn in runners.values():
        for _ in range(warmup):
            fn(X)

    times: Dict[str, List[float]] = {name: [] for name in runners}
    names = list(runners.keys())

    for i in range(repeats):
        order = names if i % 2 == 0 else names[::-1]
        for name in order:
            t0 = time.perf_counter()
            runners[name](X)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            times[name].append(elapsed_ms)

    stats: Dict[str, Dict[str, float]] = {}
    for name, measurements in times.items():
        stats[name] = {
            "mean": float(statistics.mean(measurements)),
            "std": float(statistics.pstdev(measurements)),
            "median": float(statistics.median(measurements)),
            "min": float(min(measurements)),
            "max": float(max(measurements)),
        }
    return stats


def measure_wine_energy(
    runners: Dict[str, Callable[[np.ndarray], Tuple[np.ndarray, int]]],
    X: np.ndarray,
    repeats_per_trial: int = 2000,
    trials: int = 3,
    output_dir: str = "results",
    emissions_file: str = "wine_benchmark_emissions.csv",
) -> Dict[str, Dict[str, Any]]:
    """
    Measures CodeCarbon emissions across batched Wine trials with explicit limitation logging.
    """
    results: Dict[str, Dict[str, Any]] = {}
    if not CODECARBON_AVAILABLE:
        for name in runners:
            results[name] = {
                "energy_kwh": float("nan"),
                "emissions_gco2e": float("nan"),
                "sensor_type": "CodeCarbon unavailable",
                "hardware_rapl_detected": False,
            }
        return results

    os.makedirs(output_dir, exist_ok=True)
    trial_log_path = os.path.join(output_dir, emissions_file)
    if os.path.exists(trial_log_path):
        try:
            os.remove(trial_log_path)
        except OSError:
            pass

    energy_records: Dict[str, List[float]] = {name: [] for name in runners}
    emission_records: Dict[str, List[float]] = {name: [] for name in runners}
    hardware_rapl_detected = False

    names = list(runners.keys())
    for trial in range(trials):
        order = names if trial % 2 == 0 else names[::-1]
        for name in order:
            fn = runners[name]
            tracker = OfflineEmissionsTracker(
                project_name=f"GreenAI_Wine_{name.replace(' ', '_')}_t{trial}",
                country_iso_code="IND",
                output_dir=output_dir,
                output_file=emissions_file,
                log_level="error",
                measure_power_secs=1,
                save_to_file=True,
            )
            tracker.start()
            try:
                for _ in range(repeats_per_trial):
                    fn(X)
            finally:
                tracker.stop()

            data = getattr(tracker, "final_emissions_data", None)
            energy = getattr(data, "energy_consumed", None)
            emissions = getattr(data, "emissions", None)
            cpu_power = getattr(data, "cpu_power", 0.0)

            if cpu_power and float(cpu_power) > 0.0:
                hardware_rapl_detected = True

            energy_records[name].append(float(energy) if energy is not None else float("nan"))
            emission_records[name].append(float(emissions * 1000.0) if emissions is not None else float("nan"))

    for name in runners:
        valid_energies = [e for e in energy_records[name] if not np.isnan(e)]
        valid_emissions = [m for m in emission_records[name] if not np.isnan(m)]

        med_energy = statistics.median(valid_energies) if valid_energies else float("nan")
        med_emissions = statistics.median(valid_emissions) if valid_emissions else float("nan")

        results[name] = {
            "energy_kwh": med_energy,
            "emissions_gco2e": med_emissions,
            "sensor_type": "Hardware RAPL" if hardware_rapl_detected else "TDP/Heuristic Estimate",
            "hardware_rapl_detected": hardware_rapl_detected,
        }

    return results


def run_wine_experiment(
    repeats: int = 150,
    warmup: int = 20,
    include_energy: bool = True,
    energy_repeats: int = 2000,
    energy_trials: int = 3,
    models_dir: str = "models_wine",
    output_dir: str = "results",
    output_filename: str = "wine_benchmark.csv",
    camas_lambda: float = 0.08,
    retrain: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes the full Wine generalization benchmark across all five methods.
    """
    # 1. Load existing Wine models if present, or train if missing
    model_files = ["decision_tree.pkl", "neural_network.pkl", "router.pkl", "scaler.pkl"]
    has_models = all(os.path.exists(os.path.join(models_dir, f)) for f in model_files)

    if has_models and not retrain:
        tree, neural_net, router, scaler = load_wine_models(models_dir=models_dir)
    else:
        tree, neural_net, router, scaler = train_wine_models(models_dir=models_dir)

    # 2. Load dataset
    _, X_val, X_test, _, y_val, y_test, _, _, _ = get_wine_dataset(scale=True, random_state=42)

    # 3. Calibrate CAMAS strictly on validation partition
    camas = CostAwareAdaptiveSelector(tree, neural_net)
    camas.fit_policy(X_val, y_val, lambda_penalty=camas_lambda)

    # 4. Build runners
    runners = build_wine_runners(tree, neural_net, router, camas, cascade_threshold=0.80)

    total_test_samples = len(y_test)
    pred_data = {}
    for name, fn in runners.items():
        preds, nn_count = fn(X_test)
        acc = float(accuracy_score(y_test, preds) * 100.0)
        complexity = compute_wine_theoretical_complexity(name, nn_count, total_test_samples)
        pred_data[name] = {
            "predictions": preds,
            "accuracy": acc,
            "nn_count": nn_count,
            "nn_usage_pct": (nn_count / total_test_samples) * 100.0,
            "complexity": complexity,
        }

    # 5. Measure latency distribution
    latency_stats = measure_wine_latency(runners, X_test, repeats=repeats, warmup=warmup)

    # 6. Energy profiling
    energy_stats = {}
    if include_energy:
        energy_stats = measure_wine_energy(
            runners,
            X_test,
            repeats_per_trial=energy_repeats,
            trials=energy_trials,
            output_dir=output_dir,
            emissions_file="wine_benchmark_emissions.csv",
        )
    else:
        for name in runners:
            energy_stats[name] = {
                "energy_kwh": float("nan"),
                "emissions_gco2e": float("nan"),
                "sensor_type": "Energy profiling skipped",
                "hardware_rapl_detected": False,
            }

    # 7. Assemble benchmark table
    rows = []
    nn_latency_median = latency_stats["Neural Network"]["median"]
    nn_accuracy = pred_data["Neural Network"]["accuracy"]

    for name in runners:
        lat = latency_stats[name]
        pdata = pred_data[name]
        en = energy_stats[name]

        time_saved_vs_nn = ((1.0 - (lat["median"] / nn_latency_median)) * 100.0)
        acc_kept_vs_nn = (pdata["accuracy"] / nn_accuracy) * 100.0

        rows.append({
            "Method": name,
            "Accuracy (%)": round(pdata["accuracy"], 2),
            "Test Samples": total_test_samples,
            "NN Invocations": pdata["nn_count"],
            "NN Usage (%)": round(pdata["nn_usage_pct"], 1),
            "Latency Mean (ms)": round(lat["mean"], 4),
            "Latency Std (ms)": round(lat["std"], 4),
            "Latency Median (ms)": round(lat["median"], 4),
            "Latency Min (ms)": round(lat["min"], 4),
            "Latency Max (ms)": round(lat["max"], 4),
            "Latency Saved vs NN (%)": round(time_saved_vs_nn, 1),
            "Accuracy Kept vs NN (%)": round(acc_kept_vs_nn, 1),
            "Theoretical FLOPs/Sample": int(pdata["complexity"]["flops_per_sample"]),
            "Theoretical FLOPs Saved (%)": round(pdata["complexity"]["flops_reduction_pct"], 1),
            "CodeCarbon Energy (kWh)": en["energy_kwh"],
            "CodeCarbon Emissions (gCO2e)": en["emissions_gco2e"],
            "Energy Sensor Status": en["sensor_type"],
        })

    df = pd.DataFrame(rows)

    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, output_filename)
    df.to_csv(out_csv, index=False)

    summary = {
        "output_csv": out_csv,
        "test_samples": total_test_samples,
        "camas_threshold": camas.threshold_,
        "camas_lambda": camas_lambda,
        "limitations": WINE_LIMITATIONS,
    }
    return df, summary


def main():
    parser = argparse.ArgumentParser(description="Wine Generalization Benchmark for Green AI")
    parser.add_argument("--repeats", type=int, default=150, help="Latency repetitions")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--skip-energy", action="store_true", help="Skip CodeCarbon energy tracking")
    parser.add_argument("--energy-repeats", type=int, default=2000, help="Repeats per energy trial")
    parser.add_argument("--energy-trials", type=int, default=3, help="Energy trials")
    parser.add_argument("--output-file", type=str, default="wine_benchmark.csv", help="Output CSV name")
    parser.add_argument("--camas-lambda", type=float, default=0.08, help="CAMAS lambda parameter")
    args = parser.parse_args()

    print("\n=======================================================")
    print("  WINE GENERALIZATION BENCHMARK (SECOND DATASET)       ")
    print("=======================================================")
    print(f"Dataset: Wine (178 samples, 13 features, 3 classes)")
    print(f"Test Set: 36 held-out samples | Repetitions: {args.repeats}")
    print(f"CAMAS Green Penalty (lambda): {args.camas_lambda}")

    df, summary = run_wine_experiment(
        repeats=args.repeats,
        warmup=args.warmup,
        include_energy=(not args.skip_energy),
        energy_repeats=args.energy_repeats,
        energy_trials=args.energy_trials,
        output_filename=args.output_file,
        camas_lambda=args.camas_lambda,
    )

    print("\n--- WINE BENCHMARK RESULTS (Held-Out Test Set: 36 samples) ---")
    cols = [
        "Method", "Accuracy (%)", "NN Usage (%)",
        "Latency Median (ms)", "Latency Std (ms)", "Latency Saved vs NN (%)",
        "Theoretical FLOPs Saved (%)", "Energy Sensor Status"
    ]
    print(df[cols].to_string(index=False))

    print(f"\nSaved Wine benchmark to: {summary['output_csv']}")
    print(f"CAMAS calibrated threshold on Wine: {summary['camas_threshold']:.4f}")

    print("\n--- GENERALIZATION & MEASUREMENT LIMITATIONS ---")
    for k, v in summary["limitations"].items():
        print(f"• {k}: {v}")


if __name__ == "__main__":
    main()
