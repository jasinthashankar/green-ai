
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# Allow imports from the existing src folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import get_dataset
from cascading import predict_with_cascade


def main():
    # Load the existing trained models.
    tree = joblib.load(PROJECT_ROOT / "models" / "decision_tree.pkl")
    neural_net = joblib.load(
        PROJECT_ROOT / "models" / "neural_network.pkl"
    )

    # Use validation data to compare thresholds.
    _, X_val, _, _, y_val, _ = get_dataset()

    thresholds = [0.60, 0.70, 0.80, 0.90, 0.95]
    timing_repeats = 5
    results = []

    for threshold in thresholds:
        # Warm up the models before measuring latency.
        predict_with_cascade(
            tree, neural_net, X_val[:20], threshold
        )

        run_times = []
        predictions = None
        escalated = 0

        for _ in range(timing_repeats):
            preds, escalated, elapsed = predict_with_cascade(
                tree,
                neural_net,
                X_val,
                confidence_threshold=threshold,
            )

            run_times.append(elapsed)
            predictions = preds

        accuracy = accuracy_score(y_val, predictions)
        escalation_rate = escalated / len(X_val) * 100
        median_total_ms = float(np.median(run_times) * 1000)
        median_per_sample_ms = median_total_ms / len(X_val)

        results.append({
            "Threshold": threshold,
            "Validation Accuracy (%)": accuracy * 100,
            "NN Escalation (%)": escalation_rate,
            "NN Escalated Samples": escalated,
            "Median Total Time (ms)": median_total_ms,
            "Median Time per Sample (ms)": median_per_sample_ms,
        })

        print(
            f"Threshold {threshold:.2f} | "
            f"Accuracy {accuracy:.2%} | "
            f"NN usage {escalation_rate:.2f}% | "
            f"Median total {median_total_ms:.3f} ms"
        )

    results_df = pd.DataFrame(results)

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    output_path = results_dir / "threshold_experiment.csv"
    results_df.to_csv(output_path, index=False)

    print("\n========== THRESHOLD EXPERIMENT ==========")
    print(results_df.to_string(index=False))
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
