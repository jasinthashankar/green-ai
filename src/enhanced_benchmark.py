"""
Enhanced Multi-Tier Benchmark for Green AI Dynamic Model Selection.

Motivated by the multi-tier evaluation principles and measurement uncertainty guidelines in:
'Green Artificial Intelligence: A Comprehensive Review of Metrics, Tools, Challenges,
Trends, and Future Prospects' (2026), Springer, DOI: 10.1007/s11831-026-10546-2.

This module compares five methods on the identical held-out test partition:
1. Decision Tree (Depth 5 baseline)
2. Neural Network (MLP 60,570 parameters ceiling)
3. Dynamic Cascade (0.80 confidence threshold)
4. Dynamic Routing (Logistic Regression on tree probabilities)
5. CAMAS (Cost-Aware Multi-Criteria Adaptive Selection)

Preserves all pretrained models without retraining.
Uses validation data only for policy calibration.
Reports latency dispersion (mean, std, median, min, max), theoretical FLOPs,
and CodeCarbon energy with explicit documentation of profiler limitations.
"""

import argparse
import os
import sys
import time
import statistics
from typing import Callable, Dict, List, Tuple, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# Ensure import compatibility regardless of execution working directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from dataset import get_dataset
    from adaptive_selector import CostAwareAdaptiveSelector
except ImportError:
    from src.dataset import get_dataset
    from src.adaptive_selector import CostAwareAdaptiveSelector

try:
    from codecarbon import OfflineEmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False


# Documented measurement limitations based on the 2026 Springer Green AI Review Paper
MEASUREMENT_LIMITATIONS = {
    "software_tracker": "CodeCarbon OfflineEmissionsTracker (country ISO: IND)",
    "sampling_frequency": "1 Hz (measure_power_secs=1.0)",
    "sensor_status": (
        "On Windows without direct hardware RAPL / NVML sensor drivers, "
        "CodeCarbon utilizes a constant TDP / CPU package heuristic. "
        "Energy measurements scale proportionally with execution runtime."
    ),
    "micro_inference_uncertainty": (
        "Single-batch inference (360 samples) takes ~0.25 to 2.0 ms, which is "
        "well below the 1,000 ms polling resolution of software emissions trackers. "
        "Energy must be aggregated over synthetic repeated loops to capture measurable intervals."
    ),
    "flops_vs_energy_distinction": (
        "Theoretical FLOPs calculate algorithmic operations and do not directly equate "
        "to hardware Joules due to cache hierarchies, memory bandwidth, and SIMD efficiency."
    ),
}


def load_pretrained_assets(
    models_dir: str = "models"
) -> Tuple[Any, Any, Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads existing pretrained models and the stratified dataset.
    Strictly read-only; does not modify or retrain existing .pkl files.
    """
    tree_path = os.path.join(models_dir, "decision_tree.pkl")
    nn_path = os.path.join(models_dir, "neural_network.pkl")
    router_path = os.path.join(models_dir, "router.pkl")

    if not all(os.path.exists(p) for p in [tree_path, nn_path, router_path]):
        raise FileNotFoundError(
            f"Pretrained model files missing. Checked: {tree_path}, {nn_path}, {router_path}"
        )

    tree = joblib.load(tree_path)
    neural_net = joblib.load(nn_path)
    router = joblib.load(router_path)

    X_train, X_val, X_test, y_train, y_val, y_test = get_dataset()
    return tree, neural_net, router, X_train, X_val, X_test, y_train, y_val, y_test


def get_calibrated_camas(
    tree: Any,
    neural_net: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lambda_penalty: float = 0.20,
) -> CostAwareAdaptiveSelector:
    """
    Calibrates the CAMAS adaptive selection policy strictly on validation data.
    Preserves data hygiene by never using test labels for tuning.
    """
    selector = CostAwareAdaptiveSelector(tree, neural_net)
    selector.fit_policy(X_val, y_val, lambda_penalty=lambda_penalty)
    return selector


def build_method_runners(
    tree: Any,
    neural_net: Any,
    router: Any,
    camas_selector: CostAwareAdaptiveSelector,
    cascade_threshold: float = 0.80,
) -> Dict[str, Callable[[np.ndarray], Tuple[np.ndarray, int]]]:
    """
    Builds standardized runner functions for all five evaluated methods.
    Each runner accepts input array X and returns (predictions, nn_invocation_count).
    All methods operate in batch mode to ensure execution fairness.
    """
    def run_decision_tree(X: np.ndarray) -> Tuple[np.ndarray, int]:
        return tree.predict(X), 0

    def run_neural_network(X: np.ndarray) -> Tuple[np.ndarray, int]:
        return neural_net.predict(X), len(X)

    def run_dynamic_cascade(X: np.ndarray) -> Tuple[np.ndarray, int]:
        tree_probs = tree.predict_proba(X)
        pred = tree.classes_[tree_probs.argmax(axis=1)]
        hard_mask = tree_probs.max(axis=1) < cascade_threshold
        hard_count = int(hard_mask.sum())
        if hard_count > 0:
            pred[hard_mask] = neural_net.predict(X[hard_mask])
        return pred, hard_count

    def run_dynamic_routing(X: np.ndarray) -> Tuple[np.ndarray, int]:
        tree_probs = tree.predict_proba(X)
        pred = tree.classes_[tree_probs.argmax(axis=1)]
        use_nn_mask = (router.predict(tree_probs) == 1)
        nn_count = int(use_nn_mask.sum())
        if nn_count > 0:
            pred[use_nn_mask] = neural_net.predict(X[use_nn_mask])
        return pred, nn_count

    def run_camas(X: np.ndarray) -> Tuple[np.ndarray, int]:
        pred, nn_count, _ = camas_selector.predict(X)
        return pred, int(nn_count)

    return {
        "Decision Tree": run_decision_tree,
        "Neural Network": run_neural_network,
        "Dynamic Cascade": run_dynamic_cascade,
        "Dynamic Routing": run_dynamic_routing,
        "CAMAS (Adaptive Selection)": run_camas,
    }


def compute_theoretical_complexity(
    method_name: str,
    nn_count: int,
    total_samples: int
) -> Dict[str, float]:
    """
    Calculates theoretical computational operations based on architecture definitions:
    - Decision Tree (depth 5): 5 scalar comparisons per sample.
    - Neural Network MLP (64->256->128->64->32->16->10):
      60,064 Multiply-Accumulate operations = 120,128 FLOPs per sample.
    - Dynamic Cascade: Tree comparisons (5) + NN FLOPs on escalated samples.
    - Dynamic Routing: Tree (5) + Logistic Regression router (20 FLOPs) + NN on routed samples.
    - CAMAS: Tree (5) + Multi-criteria uncertainty evaluator (45 FLOPs) + NN on escalated samples.
    """
    tree_flops_per_sample = 5.0
    nn_flops_per_sample = 120128.0

    if method_name == "Decision Tree":
        total_flops = tree_flops_per_sample * total_samples
    elif method_name == "Neural Network":
        total_flops = nn_flops_per_sample * total_samples
    elif method_name == "Dynamic Cascade":
        total_flops = (tree_flops_per_sample * total_samples) + (nn_count * nn_flops_per_sample)
    elif method_name == "Dynamic Routing":
        router_overhead_flops = 20.0 * total_samples
        total_flops = (tree_flops_per_sample * total_samples) + router_overhead_flops + (nn_count * nn_flops_per_sample)
    elif "CAMAS" in method_name:
        camas_overhead_flops = 45.0 * total_samples
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


def measure_latency_distribution(
    runners: Dict[str, Callable[[np.ndarray], Tuple[np.ndarray, int]]],
    X: np.ndarray,
    repeats: int = 150,
    warmup: int = 20,
) -> Dict[str, Dict[str, float]]:
    """
    Measures latency with warm-up cycles and interleaved execution order.
    Alternating execution order mitigates CPU thermal throttling and cache bias.
    Reports: mean, std, median, min, max (in milliseconds).
    """
    # 1. Warm-up calls
    for fn in runners.values():
        for _ in range(warmup):
            fn(X)

    # 2. Repeated measurements with alternated order
    times: Dict[str, List[float]] = {name: [] for name in runners}
    names = list(runners.keys())

    for i in range(repeats):
        order = names if i % 2 == 0 else names[::-1]
        for name in order:
            t0 = time.perf_counter()
            runners[name](X)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            times[name].append(elapsed_ms)

    # 3. Compute statistical distribution
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


def measure_codecarbon_energy(
    runners: Dict[str, Callable[[np.ndarray], Tuple[np.ndarray, int]]],
    X: np.ndarray,
    repeats_per_trial: int = 1000,
    trials: int = 3,
    output_dir: str = "results",
    emissions_file: str = "enhanced_benchmark_emissions.csv",
) -> Dict[str, Dict[str, Any]]:
    """
    Profiles energy and operational carbon emissions using CodeCarbon.
    Runs multiple trials of batched repetitions to aggregate sufficient runtime
    for the 1 Hz software tracker.

    Strictly logs sensor status and flags constant-TDP heuristic estimations.
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

    # Clean existing trial log for this benchmark run so historical trials are not mixed
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
                project_name=f"GreenAI_Enhanced_{name.replace(' ', '_')}_t{trial}",
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


def run_enhanced_benchmark(
    repeats: int = 150,
    warmup: int = 20,
    include_energy: bool = True,
    energy_repeats: int = 1000,
    energy_trials: int = 3,
    output_dir: str = "results",
    output_filename: str = "enhanced_benchmark.csv",
    camas_lambda: float = 0.20,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes the fair five-method benchmark on the held-out test set.
    Saves outputs to distinct new filenames without modifying existing files.
    """
    # 1. Load models and dataset
    tree, neural_net, router, _, X_val, X_test, _, y_val, y_test = load_pretrained_assets()

    # 2. Calibrate CAMAS strictly on validation data
    camas = get_calibrated_camas(tree, neural_net, X_val, y_val, lambda_penalty=camas_lambda)

    # 3. Build runners
    runners = build_method_runners(tree, neural_net, router, camas, cascade_threshold=0.80)

    # 4. Measure predictions and accuracy once outside timing loops
    pred_data = {}
    total_test_samples = len(y_test)
    for name, fn in runners.items():
        preds, nn_count = fn(X_test)
        acc = float(accuracy_score(y_test, preds) * 100.0)
        complexity = compute_theoretical_complexity(name, nn_count, total_test_samples)
        pred_data[name] = {
            "predictions": preds,
            "accuracy": acc,
            "nn_count": nn_count,
            "nn_usage_pct": (nn_count / total_test_samples) * 100.0,
            "complexity": complexity,
        }

    # 5. Measure latency distribution
    latency_stats = measure_latency_distribution(runners, X_test, repeats=repeats, warmup=warmup)

    # 6. Profile energy/emissions with CodeCarbon
    energy_stats = {}
    if include_energy:
        energy_stats = measure_codecarbon_energy(
            runners,
            X_test,
            repeats_per_trial=energy_repeats,
            trials=energy_trials,
            output_dir=output_dir,
            emissions_file="enhanced_benchmark_emissions.csv",
        )
    else:
        for name in runners:
            energy_stats[name] = {
                "energy_kwh": float("nan"),
                "emissions_gco2e": float("nan"),
                "sensor_type": "Energy profiling skipped",
                "hardware_rapl_detected": False,
            }

    # 7. Assemble comprehensive multi-tier benchmark table
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

    # 8. Save output table to distinct file in results/
    os.makedirs(output_dir, exist_ok=True)
    out_csv_path = os.path.join(output_dir, output_filename)
    df.to_csv(out_csv_path, index=False)

    summary = {
        "output_csv": out_csv_path,
        "repeats": repeats,
        "test_samples": total_test_samples,
        "camas_calibrated_threshold": camas.threshold_,
        "camas_lambda": camas_lambda,
        "limitations": MEASUREMENT_LIMITATIONS,
    }

    return df, summary


def main():
    parser = argparse.ArgumentParser(description="Enhanced Green AI Multi-Tier Benchmark")
    parser.add_argument("--repeats", type=int, default=150, help="Latency repetitions")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--skip-energy", action="store_true", help="Skip CodeCarbon energy tracking")
    parser.add_argument("--energy-repeats", type=int, default=1000, help="CodeCarbon repeats per trial")
    parser.add_argument("--energy-trials", type=int, default=3, help="CodeCarbon trials")
    parser.add_argument("--output-file", type=str, default="enhanced_benchmark.csv", help="Output CSV name")
    parser.add_argument("--camas-lambda", type=float, default=0.20, help="Green penalty parameter lambda")
    args = parser.parse_args()

    print("\n=======================================================")
    print("  ENHANCED GREEN AI MULTI-TIER BENCHMARK (STAGE 2)     ")
    print("=======================================================")
    print(f"Test Set: 360 held-out samples | Repetitions: {args.repeats}")
    print(f"CAMAS Green Penalty (lambda): {args.camas_lambda}")
    print("Calibrating CAMAS strictly on validation data...")

    df, summary = run_enhanced_benchmark(
        repeats=args.repeats,
        warmup=args.warmup,
        include_energy=(not args.skip_energy),
        energy_repeats=args.energy_repeats,
        energy_trials=args.energy_trials,
        output_filename=args.output_file,
        camas_lambda=args.camas_lambda,
    )

    print("\n--- BENCHMARK RESULTS (Held-Out Test Set) ---")
    cols_to_print = [
        "Method", "Accuracy (%)", "NN Usage (%)",
        "Latency Median (ms)", "Latency Std (ms)", "Latency Saved vs NN (%)",
        "Theoretical FLOPs Saved (%)", "Energy Sensor Status"
    ]
    print(df[cols_to_print].to_string(index=False))

    print(f"\nSaved comprehensive benchmark to: {summary['output_csv']}")
    print(f"CAMAS calibrated threshold: {summary['camas_calibrated_threshold']:.4f}")

    print("\n--- MEASUREMENT LIMITATIONS & UNCERTAINTY NOTICE ---")
    for k, v in summary["limitations"].items():
        print(f"• {k}: {v}")


if __name__ == "__main__":
    main()
