
import os
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from dataset import get_dataset


# Load existing models without retraining them
tree = joblib.load("models/decision_tree.pkl")
neural_net = joblib.load("models/neural_network.pkl")

# Keep the existing train/validation/test split
_, X_val, X_test, _, y_val, y_test = get_dataset()

# Split validation data into router-training and threshold-tuning parts
from sklearn.model_selection import train_test_split

X_router_train, X_tune, y_router_train, y_tune = train_test_split(
    X_val,
    y_val,
    test_size=0.50,
    random_state=42,
    stratify=y_val,
)

# Build router-training labels using only the router-training partition
tree_train_probs = tree.predict_proba(X_router_train)
tree_train_preds = tree.predict(X_router_train)
nn_train_preds = neural_net.predict(X_router_train)

router_labels = (
    (tree_train_preds != y_router_train)
    & (nn_train_preds == y_router_train)
).astype(int)

# Train a temporary router; do not overwrite models/router.pkl
experiment_router = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42,
)
experiment_router.fit(tree_train_probs, router_labels)

# Tune thresholds on a separate validation partition
tree_tune_probs = tree.predict_proba(X_tune)
tune_router_probs = experiment_router.predict_proba(tree_tune_probs)[:, 1]
tree_tune_preds = tree.predict(X_tune)
nn_tune_preds = neural_net.predict(X_tune)

results = []

for threshold in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
    use_nn = tune_router_probs >= threshold

    predictions = tree_tune_preds.copy()
    predictions[use_nn] = nn_tune_preds[use_nn]

    results.append({
        "threshold": threshold,
        "accuracy_percent": round(
            accuracy_score(y_tune, predictions) * 100, 2
        ),
        "neural_network_usage_percent": round(
            use_nn.mean() * 100, 2
        ),
        "nn_samples": int(use_nn.sum()),
        "total_samples": len(y_tune),
    })

results_df = pd.DataFrame(results)

os.makedirs("results", exist_ok=True)
output_path = "results/router_threshold_experiment.csv"
results_df.to_csv(output_path, index=False)

print("\nROUTER THRESHOLD EXPERIMENT")
print("-" * 72)
print(f"Router-training samples: {len(X_router_train)}")
print(f"Threshold-tuning samples: {len(X_tune)}")
print("\nResults:")
print(results_df.to_string(index=False))
print(f"\nSaved results to: {output_path}")
print("\nExisting models/router.pkl was not modified.")
print("The test set was not used for threshold selection.")
