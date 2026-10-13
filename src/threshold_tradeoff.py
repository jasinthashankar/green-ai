"""Accuracy vs neural-network usage as the cascade threshold changes."""
import os
import time
import statistics
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from dataset import get_dataset

tree = joblib.load("models/decision_tree.pkl")
nn = joblib.load("models/neural_network.pkl")
_, X_val, X_test, _, y_val, y_test = get_dataset()

THRESHOLDS = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
REPEATS = 200


def cascade(X, thr):
    proba = tree.predict_proba(X)
    pred = tree.classes_[proba.argmax(axis=1)]
    hard = proba.max(axis=1) < thr
    if hard.any():
        pred[hard] = nn.predict(X[hard])
    return pred, hard.mean()


def median_time(fn):
    for _ in range(20):
        fn()
    t = []
    for _ in range(REPEATS):
        s = time.perf_counter(); fn(); t.append((time.perf_counter() - s) * 1000)
    return statistics.median(t)


nn_time = median_time(lambda: nn.predict(X_test))
nn_acc = accuracy_score(y_test, nn.predict(X_test)) * 100

rows = []
for thr in THRESHOLDS:
    pv, _ = cascade(X_val, thr)                 # choose settings on validation
    pt, usage = cascade(X_test, thr)            # report on test
    ms = median_time(lambda: cascade(X_test, thr))
    rows.append({
        "Threshold": thr,
        "Val accuracy (%)": round(accuracy_score(y_val, pv) * 100, 2),
        "Test accuracy (%)": round(accuracy_score(y_test, pt) * 100, 2),
        "NN usage (%)": round(float(usage) * 100, 1),
        "Time (ms)": round(ms, 3),
        "Time saved vs NN (%)": round((1 - ms / nn_time) * 100, 1),
        "Accuracy kept (%)": round(accuracy_score(y_test, pt) * 100 / nn_acc * 100, 1),
    })

df = pd.DataFrame(rows)
print(df.to_string(index=False))
os.makedirs("results", exist_ok=True)
df.to_csv("results/threshold_tradeoff.csv", index=False)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(df["NN usage (%)"], df["Test accuracy (%)"], marker="o")
for _, r in df.iterrows():
    ax[0].annotate(str(r["Threshold"]), (r["NN usage (%)"], r["Test accuracy (%)"]),
                   textcoords="offset points", xytext=(4, 4), fontsize=8)
ax[0].set_xlabel("Neural network usage (%)"); ax[0].set_ylabel("Test accuracy (%)")
ax[0].set_title("Accuracy vs NN usage"); ax[0].grid(alpha=0.3)
ax[1].plot(df["Threshold"], df["Time saved vs NN (%)"], marker="s", color="green")
ax[1].axhline(0, color="gray", linewidth=0.8)
ax[1].set_xlabel("Confidence threshold"); ax[1].set_ylabel("Time saved vs NN (%)")
ax[1].set_title("Time saved vs threshold"); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig("results/threshold_tradeoff.png", dpi=150)
print("Saved results/threshold_tradeoff.csv and .png")