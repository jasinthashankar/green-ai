"""
Model-size study: how much does dynamic selection save as the
neural network gets bigger? Uses the same decision tree for all sizes.
"""
import os
import time
import statistics
import joblib
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from dataset import get_dataset

THRESHOLD = 0.80
REPEATS = 200

SIZES = {
    "Small (32)":               (32,),
    "Medium (128,64)":          (128, 64),
    "Large (256,128,64,32,16)": (256, 128, 64, 32, 16),
    "XL (512,512,256)":         (512, 512, 256),
    "XXL (1024,1024,512)":      (1024, 1024, 512),
}

X_train, X_val, X_test, y_train, y_val, y_test = get_dataset()
tree = joblib.load("models/decision_tree.pkl")


def make_methods(nn, router):
    def run_nn(X):
        return nn.predict(X), 1.0

    def run_cascade(X):
        proba = tree.predict_proba(X)
        pred = tree.classes_[proba.argmax(axis=1)]
        hard = proba.max(axis=1) < THRESHOLD
        if hard.any():
            pred[hard] = nn.predict(X[hard])
        return pred, hard.mean()

    def run_routing(X):
        proba = tree.predict_proba(X)
        pred = tree.classes_[proba.argmax(axis=1)]
        use_nn = router.predict(proba) == 1
        if use_nn.any():
            pred[use_nn] = nn.predict(X[use_nn])
        return pred, use_nn.mean()

    return {"nn": run_nn, "cascade": run_cascade, "routing": run_routing}


rows = []
for label, layers in SIZES.items():
    print(f"Training {label} ...", flush=True)
    nn = MLPClassifier(hidden_layer_sizes=layers, max_iter=300,
                       random_state=42).fit(X_train, y_train)

    # Router labels: 1 = tree wrong but NN right (validation set only)
    tp, npred = tree.predict(X_val), nn.predict(X_val)
    labels = ((tp != y_val) & (npred == y_val)).astype(int)
    router = LogisticRegression(class_weight="balanced", max_iter=1000)
    router.fit(tree.predict_proba(X_val), labels)

    methods = make_methods(nn, router)
    for fn in methods.values():
        for _ in range(20):
            fn(X_test)

    times = {k: [] for k in methods}
    names = list(methods)
    for i in range(REPEATS):
        order = names if i % 2 == 0 else names[::-1]
        for k in order:
            t0 = time.perf_counter()
            methods[k](X_test)
            times[k].append((time.perf_counter() - t0) * 1000)

    med = {k: statistics.median(v) for k, v in times.items()}
    acc = {k: accuracy_score(y_test, methods[k](X_test)[0]) * 100 for k in methods}
    params = sum(w.size for w in nn.coefs_) + sum(b.size for b in nn.intercepts_)

    rows.append({
        "NN size": label,
        "Parameters": params,
        "NN accuracy (%)": round(acc["nn"], 2),
        "NN time (ms)": round(med["nn"], 3),
        "Cascade accuracy (%)": round(acc["cascade"], 2),
        "Cascade time (ms)": round(med["cascade"], 3),
        "Cascade time saved (%)": round((1 - med["cascade"] / med["nn"]) * 100, 1),
        "Routing accuracy (%)": round(acc["routing"], 2),
        "Routing time (ms)": round(med["routing"], 3),
        "Routing time saved (%)": round((1 - med["routing"] / med["nn"]) * 100, 1),
    })

df = pd.DataFrame(rows)
print()
print(df.to_string(index=False))
os.makedirs("results", exist_ok=True)
df.to_csv("results/model_size_study.csv", index=False)
print("\nSaved to results/model_size_study.csv")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 4.5))
    plt.plot(df["Parameters"], df["Cascade time saved (%)"], marker="o", label="Cascade")
    plt.plot(df["Parameters"], df["Routing time saved (%)"], marker="s", label="Routing")
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.xscale("log")
    plt.xlabel("Neural network parameters (log scale)")
    plt.ylabel("Time saved vs neural network (%)")
    plt.title("When does dynamic model selection pay off?")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("results/model_size_study.png", dpi=150)
    print("Saved results/model_size_study.png")
except Exception as e:
    print("Plot skipped:", e)