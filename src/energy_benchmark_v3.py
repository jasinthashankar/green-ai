import os
import time
import csv
import statistics
import joblib
import numpy as np

from codecarbon import OfflineEmissionsTracker
from dataset import get_dataset
from sklearn.metrics import accuracy_score

os.makedirs("results", exist_ok=True)

# Remove an old v3 emissions log so runs are not mixed together.
EMISSIONS_FILE = "energy_benchmark_v3_emissions.csv"
old = os.path.join("results", EMISSIONS_FILE)
if os.path.exists(old):
    os.remove(old)

# Load existing models and the held-out test set.
tree = joblib.load("models/decision_tree.pkl")
neural_net = joblib.load("models/neural_network.pkl")
router = joblib.load("models/router.pkl")
_, _, X_test, _, _, y_test = get_dataset()

THRESHOLD = 0.80
REPEATS = 10000
TRIALS = 5


def tree_predict_with_confidence(X):
    probabilities = tree.predict_proba(X)
    predictions = tree.classes_[np.argmax(probabilities, axis=1)]
    return probabilities, predictions


def predict_tree():
    return tree.predict(X_test)


def predict_neural_network():
    return neural_net.predict(X_test)


def predict_cascade():
    probabilities, predictions = tree_predict_with_confidence(X_test)
    confidence = np.max(probabilities, axis=1)
    indices = np.where(confidence < THRESHOLD)[0]
    if len(indices) > 0:
        predictions[indices] = neural_net.predict(X_test[indices])
    return predictions


def predict_routing():
    probabilities, predictions = tree_predict_with_confidence(X_test)
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

# Accuracy is measured once, outside the timed loops.
accuracies = {
    name: round(accuracy_score(y_test, fn()) * 100, 2)
    for name, fn in methods.items()
}

# Warm-up so first-call overhead is not timed.
for fn in methods.values():
    for _ in range(10):
        fn()

runtimes = {n: [] for n in methods}
energies = {n: [] for n in methods}

names = list(methods)
for trial in range(TRIALS):
    # Alternate the order each trial so no method always runs first or last.
    order = names if trial % 2 == 0 else names[::-1]
    for name in order:
        fn = methods[name]
        print(f"Trial {trial + 1}/{TRIALS}: {name}")

        tracker = OfflineEmissionsTracker(
            project_name=f"GreenAI_V3_{name.replace(' ', '_')}_{trial}",
            country_iso_code="IND",
            output_dir="results",
            output_file=EMISSIONS_FILE,
            log_level="error",
            measure_power_secs=1,
        )

        tracker.start()
        t0 = time.perf_counter()
        try:
            for _ in range(REPEATS):
                fn()
        finally:
            elapsed = time.perf_counter() - t0
            tracker.stop()

        data = getattr(tracker, "final_emissions_data", None)
        energy = getattr(data, "energy_consumed", None)

        runtimes[name].append(elapsed)
        energies[name].append(float(energy) if energy is not None else float("nan"))

# Summary per method.
rows = []
nn_median_energy = statistics.median(energies["Neural Network"])
nn_median_time = statistics.median(runtimes["Neural Network"])

for name in methods:
    rt = runtimes[name]
    en = energies[name]
    med_e = statistics.median(en)
    med_t = statistics.median(rt)
    rows.append({
        "Method": name,
        "Accuracy (%)": accuracies[name],
        "Trials": TRIALS,
        "Repeats per trial": REPEATS,
        "Median runtime (s)": round(med_t, 4),
        "Runtime min (s)": round(min(rt), 4),
        "Runtime max (s)": round(max(rt), 4),
        "Median energy (kWh)": round(med_e, 12),
        "Energy min (kWh)": round(min(en), 12),
        "Energy max (kWh)": round(max(en), 12),
        "Energy vs NN (%)": round((med_e / nn_median_energy - 1) * 100, 1),
        "Runtime vs NN (%)": round((med_t / nn_median_time - 1) * 100, 1),
    })

out = "results/energy_benchmark_v3.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\nSaved:", out)
for r in rows:
    print(r["Method"], "| acc", r["Accuracy (%)"],
          "| runtime vs NN", r["Runtime vs NN (%)"], "%",
          "| energy vs NN", r["Energy vs NN (%)"], "%")