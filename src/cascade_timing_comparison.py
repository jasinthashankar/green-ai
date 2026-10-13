
import time
import statistics
import joblib
import numpy as np

from dataset import get_dataset


tree = joblib.load("models/decision_tree.pkl")
neural_net = joblib.load("models/neural_network.pkl")

_, _, X_test, _, _, y_test = get_dataset()
threshold = 0.80


def batch_cascade(X):
    probabilities = tree.predict_proba(X)
    tree_predictions = tree.predict(X)
    confidence = np.max(probabilities, axis=1)
    indices = np.flatnonzero(confidence < threshold)

    predictions = tree_predictions.copy()
    if len(indices):
        predictions[indices] = neural_net.predict(X[indices])

    return predictions


def singlepass_cascade(X):
    probabilities = tree.predict_proba(X)
    confidence = np.max(probabilities, axis=1)
    predictions = tree.classes_[np.argmax(probabilities, axis=1)]
    indices = np.flatnonzero(confidence < threshold)

    if len(indices):
        predictions[indices] = neural_net.predict(X[indices])

    return predictions


methods = {
    "Batch cascade": batch_cascade,
    "Single-pass cascade": singlepass_cascade,
}

# Warm up both implementations.
for function in methods.values():
    for _ in range(10):
        function(X_test)

# Check that both implementations produce identical predictions.
batch_predictions = batch_cascade(X_test)
single_predictions = singlepass_cascade(X_test)

print("Predictions identical:", np.array_equal(
    batch_predictions, single_predictions
))
print("Test accuracy:", round(
    np.mean(batch_predictions == y_test) * 100, 2
), "%")

# Alternate measurement order to reduce order bias.
measurements = {name: [] for name in methods}
for repeat in range(100):
    order = list(methods.items())
    if repeat % 2:
        order.reverse()

    for name, function in order:
        start = time.perf_counter()
        function(X_test)
        measurements[name].append(
            (time.perf_counter() - start) * 1000
        )

print("\nTIMING COMPARISON (100 repetitions)")
for name, times in measurements.items():
    print(f"{name}:")
    print(f"  Median: {statistics.median(times):.4f} ms")
    print(f"  Mean:   {statistics.mean(times):.4f} ms")

print("\nNo models or result files were modified.")
