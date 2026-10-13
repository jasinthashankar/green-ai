
import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from dataset import get_dataset


# Load models
tree = joblib.load("models/decision_tree.pkl")
neural_net = joblib.load("models/neural_network.pkl")
router = joblib.load("models/router.pkl")

# Load held-out test data
_, _, X_test, _, _, y_test = get_dataset()


def predict_tree(X):
    return tree.predict(X)


def predict_neural_network(X):
    return neural_net.predict(X)



def predict_cascade(X, threshold=0.80):
    # Predict all samples with the Decision Tree in one batch
    tree_probabilities = tree.predict_proba(X)
    predictions = tree.predict(X)

    # Find samples whose tree confidence is below threshold
    confidence = np.max(tree_probabilities, axis=1)
    low_confidence_indices = np.where(confidence < threshold)[0]

    # Send only those samples to the Neural Network
    if len(low_confidence_indices) > 0:
        predictions[low_confidence_indices] = neural_net.predict(
            X[low_confidence_indices]
        )

    return predictions



def predict_routing(X):
    tree_probabilities = tree.predict_proba(X)
    predictions = tree.predict(X)
    choices = router.predict(tree_probabilities)

    nn_indices = np.where(choices == 1)[0]

    if len(nn_indices) > 0:
        predictions[nn_indices] = neural_net.predict(X[nn_indices])

    return predictions


methods = {
    "Decision Tree": lambda: predict_tree(X_test),
    "Neural Network": lambda: predict_neural_network(X_test),
    "Dynamic Cascade": lambda: predict_cascade(X_test),
    "Dynamic Routing": lambda: predict_routing(X_test),
}

WARMUP_RUNS = 3
BENCHMARK_RUNS = 30

results = []

print("Warming up models...")

for name, predict_function in methods.items():
    for _ in range(WARMUP_RUNS):
        predict_function()

print("\nRunning repeated benchmarks...")

for name, predict_function in methods.items():
    durations = []
    predictions = None

    for _ in range(BENCHMARK_RUNS):
        start = time.perf_counter()
        predictions = predict_function()
        durations.append(time.perf_counter() - start)

    durations = np.array(durations)

    accuracy = accuracy_score(y_test, predictions)
    median_seconds = float(np.median(durations))
    mean_seconds = float(np.mean(durations))
    std_seconds = float(np.std(durations, ddof=1))

    results.append({
        "Method": name,
        "Accuracy (%)": accuracy * 100,
        "Median time (ms)": median_seconds * 1000,
        "Mean time (ms)": mean_seconds * 1000,
        "Std deviation (ms)": std_seconds * 1000,
        "Samples per second": (
            len(X_test) / median_seconds
            if median_seconds > 0 else float("nan")
        ),
    })

    print(
        f"{name}: accuracy={accuracy * 100:.2f}%, "
        f"median={median_seconds * 1000:.4f} ms, "
        f"mean={mean_seconds * 1000:.4f} ms"
    )

# Save results for your report and dashboard
os.makedirs("results", exist_ok=True)

df = pd.DataFrame(results)
output_path = "results/benchmark_results.csv"
df.to_csv(output_path, index=False)

print("\n--- Final Benchmark Table ---")
print(df.round(4).to_string(index=False))
print(f"\nResults saved to: {output_path}")
