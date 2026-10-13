
import os
import time
import csv
import joblib
import numpy as np

from codecarbon import EmissionsTracker
from dataset import get_dataset
from sklearn.metrics import accuracy_score


# Keep existing files unchanged.
os.makedirs("results", exist_ok=True)

# Load existing trained models and the held-out test set.
tree = joblib.load("models/decision_tree.pkl")
neural_net = joblib.load("models/neural_network.pkl")
router = joblib.load("models/router.pkl")

_, _, X_test, _, _, y_test = get_dataset()

THRESHOLD = 0.80
REPEATS = 10000


def predict_tree():
    return tree.predict(X_test)


def predict_neural_network():
    return neural_net.predict(X_test)


def predict_cascade():
    probabilities = tree.predict_proba(X_test)
    predictions = tree.predict(X_test)

    confidence = np.max(probabilities, axis=1)
    indices = np.where(confidence < THRESHOLD)[0]

    if len(indices) > 0:
        predictions[indices] = neural_net.predict(X_test[indices])

    return predictions


def predict_routing():
    probabilities = tree.predict_proba(X_test)
    predictions = tree.predict(X_test)

    choices = router.predict(probabilities)
    indices = np.where(choices == 1)[0]

    if len(indices) > 0:
        predictions[indices] = neural_net.predict(X_test[indices])

    return predictions


methods = {
    "Decision Tree": predict_tree,
    "Neural Network": predict_neural_network,
    "Dynamic Cascade": predict_cascade,
    "Dynamic Routing": predict_routing,
}

results = []

for name, predict_function in methods.items():
    print(f"\nMeasuring: {name}")

    # Warm-up runs are excluded from the timed prediction loop.
    for _ in range(10):
        predict_function()

    tracker = EmissionsTracker(
        project_name="GreenAI_V2_" + name.replace(" ", "_"),
        output_dir="results",
        output_file="energy_benchmark_v2_emissions.csv",
        log_level="error",
        save_to_file=True,
        measure_power_secs=1,
    )

    tracker.start()
    start = time.perf_counter()

    try:
        for _ in range(REPEATS):
            predictions = predict_function()
    finally:
        elapsed = time.perf_counter() - start
        emissions_kg = tracker.stop()

    accuracy = accuracy_score(y_test, predictions)
    total_samples = len(X_test) * REPEATS

    # CodeCarbon's energy_consumed is an estimate in kWh.
    final_data = getattr(tracker, "final_emissions_data", None)
    energy_kwh = getattr(final_data, "energy_consumed", None)

    row = {
        "Method": name,
        "Accuracy (%)": round(accuracy * 100, 2),
        "Test samples per run": len(X_test),
        "Repeats": REPEATS,
        "Total predictions": total_samples,
        "Runtime (seconds)": round(elapsed, 4),
        "Average batch time (ms)": round(
            elapsed * 1000 / REPEATS, 4
        ),
        "Estimated energy (kWh)": (
            round(float(energy_kwh), 12)
            if energy_kwh is not None else ""
        ),
        "Estimated emissions (kg CO2e)": (
            round(float(emissions_kg), 12)
            if emissions_kg is not None else ""
        ),
    }

    results.append(row)

    print("Accuracy:", row["Accuracy (%)"], "%")
    print("Runtime:", row["Runtime (seconds)"], "seconds")
    print("Estimated energy:", row["Estimated energy (kWh)"], "kWh")
    print("Estimated emissions:", row["Estimated emissions (kg CO2e)"],
          "kg CO2e")


output_file = "results/energy_benchmark_v2.csv"

with open(output_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\nNew results saved to:", output_file)
print("Existing benchmark and emissions CSV files were not overwritten.")
