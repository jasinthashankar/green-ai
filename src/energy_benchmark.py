import os
import time
import csv
import joblib
import numpy as np

from codecarbon import EmissionsTracker
from dataset import get_dataset


# Load trained models
tree = joblib.load("models/decision_tree.pkl")
neural_net = joblib.load("models/neural_network.pkl")
router = joblib.load("models/router.pkl")

_, _, X_test, _, _, y_test = get_dataset()

REPEATS = 2000
os.makedirs("results", exist_ok=True)


def predict_tree():
    return tree.predict(X_test)


def predict_neural_network():
    return neural_net.predict(X_test)


def predict_cascade():
    probabilities = tree.predict_proba(X_test)
    predictions = tree.predict(X_test)

    confidence = np.max(probabilities, axis=1)
    indices = np.where(confidence < 0.80)[0]

    if len(indices) > 0:
        predictions[indices] = neural_net.predict(
            X_test[indices]
        )

    return predictions


def predict_routing():
    probabilities = tree.predict_proba(X_test)
    predictions = tree.predict(X_test)

    choices = router.predict(probabilities)
    indices = np.where(choices == 1)[0]

    if len(indices) > 0:
        predictions[indices] = neural_net.predict(
            X_test[indices]
        )

    return predictions


methods = {
    "Decision Tree": predict_tree,
    "Neural Network": predict_neural_network,
    "Dynamic Cascade": predict_cascade,
    "Dynamic Routing": predict_routing,
}

results = []

for name, predict_function in methods.items():
    print(f"\nTesting {name}...")

    # Warm up the model before measuring
    for _ in range(5):
        predict_function()

    tracker = EmissionsTracker(
        project_name=name.replace(" ", "_"),
        output_dir="results",
        log_level="error",
        save_to_file=True,
    )

    tracker.start()
    start_time = time.perf_counter()

    predictions = None

    try:
        for _ in range(REPEATS):
            predictions = predict_function()
    finally:
        elapsed = time.perf_counter() - start_time
        emissions_kg = tracker.stop()

    accuracy = np.mean(predictions == y_test)
    total_samples = len(X_test) * REPEATS
    average_batch_ms = elapsed * 1000 / REPEATS

    row = {
        "Method": name,
        "Accuracy (%)": round(accuracy * 100, 2),
        "Repeats": REPEATS,
        "Total samples processed": total_samples,
        "Total runtime (seconds)": round(elapsed, 4),
        "Average batch time (ms)": round(average_batch_ms, 4),
        "Estimated emissions (kg CO2e)": emissions_kg,
    }

    results.append(row)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Total runtime: {elapsed:.4f} seconds")
    print(f"Average batch time: {average_batch_ms:.4f} ms")
    print(f"Estimated emissions: {emissions_kg} kg CO2e")


output_file = "results/energy_benchmark.csv"

with open(output_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"\nResults saved to: {output_file}")