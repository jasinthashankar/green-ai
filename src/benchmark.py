import os
import time
import statistics
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from dataset import get_dataset

THRESHOLD = 0.80
REPEATS = 300

tree = joblib.load("models/decision_tree.pkl")
nn = joblib.load("models/neural_network.pkl")
router = joblib.load("models/router.pkl")

_, _, X_test, _, _, y_test = get_dataset()


def run_tree(X):
    return tree.predict(X), 0.0


def run_nn(X):
    return nn.predict(X), 1.0


def run_cascade(X):
    proba = tree.predict_proba(X)                  # tree runs once
    pred = tree.classes_[proba.argmax(axis=1)]
    hard = proba.max(axis=1) < THRESHOLD           # low confidence
    if hard.any():
        pred[hard] = nn.predict(X[hard])           # NN only on hard samples
    return pred, hard.mean()


def run_routing(X):
    proba = tree.predict_proba(X)
    pred = tree.classes_[proba.argmax(axis=1)]
    use_nn = router.predict(proba) == 1            # router decides
    if use_nn.any():
        pred[use_nn] = nn.predict(X[use_nn])
    return pred, use_nn.mean()


methods = {
    "Decision Tree": run_tree,
    "Neural Network": run_nn,
    "Dynamic Cascade": run_cascade,
    "Dynamic Routing": run_routing,
}

# Warm-up so first-call overhead is not measured
for fn in methods.values():
    for _ in range(20):
        fn(X_test)

# Alternate the order each repeat to avoid order bias
times = {name: [] for name in methods}
names = list(methods)
for i in range(REPEATS):
    order = names if i % 2 == 0 else names[::-1]
    for name in order:
        start = time.perf_counter()
        methods[name](X_test)
        times[name].append((time.perf_counter() - start) * 1000)

rows = []
for name, fn in methods.items():
    pred, nn_usage = fn(X_test)
    rows.append({
        "Method": name,
        "Accuracy (%)": round(accuracy_score(y_test, pred) * 100, 2),
        "NN usage (%)": round(float(nn_usage) * 100, 1),
        "Median time (ms)": round(statistics.median(times[name]), 4),
        "Std (ms)": round(statistics.pstdev(times[name]), 4),
    })

df = pd.DataFrame(rows)
nn_time = df.loc[df["Method"] == "Neural Network", "Median time (ms)"].iloc[0]
nn_acc = df.loc[df["Method"] == "Neural Network", "Accuracy (%)"].iloc[0]
df["Time saved vs NN (%)"] = ((1 - df["Median time (ms)"] / nn_time) * 100).round(1)
df["Accuracy kept (%)"] = (df["Accuracy (%)"] / nn_acc * 100).round(1)

print(df.to_string(index=False))

os.makedirs("results", exist_ok=True)
df.to_csv("results/final_results.csv", index=False)
print("\nSaved to results/final_results.csv")