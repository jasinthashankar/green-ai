import joblib
import numpy as np

from dataset import get_dataset
from sklearn.metrics import accuracy_score

tree = joblib.load("models/decision_tree.pkl")
nn = joblib.load("models/neural_network.pkl")
router = joblib.load("models/router.pkl")

_, _, X_test, _, _, y_test = get_dataset()

# Get predictions from both individual models
tree_probs = tree.predict_proba(X_test)
tree_preds = tree.predict(X_test)
nn_preds = nn.predict(X_test)

# Method 1: Decision Tree only
tree_only = tree_preds.copy()

# Method 2: Neural Network only
nn_only = nn_preds.copy()

# Method 3: Dynamic Cascade (threshold = 0.80)
cascade = tree_preds.copy()
confidence = np.max(tree_probs, axis=1)
cascade_indices = np.where(confidence < 0.80)[0]
cascade[cascade_indices] = nn.predict(X_test[cascade_indices])

# Method 4: Dynamic Routing
routing = tree_preds.copy()
choices = router.predict(tree_probs)
routing_indices = np.where(choices == 1)[0]
routing[routing_indices] = nn.predict(X_test[routing_indices])

methods = [
    ("Decision Tree", tree_only, 0),
    ("Neural Network", nn_only, len(X_test)),
    ("Dynamic Cascade", cascade, len(cascade_indices)),
    ("Dynamic Routing", routing, len(routing_indices)),
]

print("\n========== TEST SET VALIDATION ==========")
print("Test samples:", len(y_test))
print("Cascade confidence threshold: 0.80")
print("-" * 65)
print(f"{'Method':<20} {'Accuracy':>12} {'NN used':>12} {'NN usage %':>12}")
print("-" * 65)

for name, predictions, nn_count in methods:
    accuracy = accuracy_score(y_test, predictions) * 100
    usage = nn_count / len(y_test) * 100
    print(f"{name:<20} {accuracy:>11.2f}% {nn_count:>12} {usage:>11.2f}%")

print("-" * 65)
print("Router choices:", dict(zip(*np.unique(choices, return_counts=True))))
print("\nValidation complete. No models were retrained.")
