
import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from dataset import get_dataset


# Load existing models. Do not overwrite saved model files.
tree = joblib.load("models/decision_tree.pkl")
nn = joblib.load("models/neural_network.pkl")

_, X_val, X_test, _, y_val, y_test = get_dataset()

# Recreate the same training/tuning split used in the experiment
X_router_train, X_tune, y_router_train, y_tune = train_test_split(
    X_val,
    y_val,
    test_size=0.50,
    random_state=42,
    stratify=y_val,
)

# Train a temporary router using only its training partition
tree_train_probs = tree.predict_proba(X_router_train)
tree_train_preds = tree.predict(X_router_train)
nn_train_preds = nn.predict(X_router_train)

router_labels = (
    (tree_train_preds != y_router_train)
    & (nn_train_preds == y_router_train)
).astype(int)

router = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42,
)
router.fit(tree_train_probs, router_labels)

# Predictions on the untouched test set
tree_probs = tree.predict_proba(X_test)
tree_preds = tree.predict(X_test)
nn_preds = nn.predict(X_test)

# Decision Tree baseline
tree_only = tree_preds.copy()

# Neural Network baseline
nn_only = nn_preds.copy()

# Existing cascade strategy, threshold 0.80
cascade = tree_preds.copy()
cascade_mask = np.max(tree_probs, axis=1) < 0.80
cascade[cascade_mask] = nn.predict(X_test[cascade_mask])

# Experimental router, selected threshold 0.40
router_scores = router.predict_proba(tree_probs)[:, 1]
router_mask = router_scores >= 0.40

routed = tree_preds.copy()
routed[router_mask] = nn.predict(X_test[router_mask])

methods = [
    ("Decision Tree", tree_only, 0),
    ("Neural Network", nn_only, len(y_test)),
    ("Dynamic Cascade", cascade, int(cascade_mask.sum())),
    ("Tuned Dynamic Routing", routed, int(router_mask.sum())),
]

print("\n========== HELD-OUT TEST RESULTS ==========")
print("Test samples:", len(y_test))
print("Experimental router threshold: 0.40")
print("-" * 72)
print(f"{'Method':<25}{'Accuracy':>12}{'NN samples':>13}{'NN usage':>13}")
print("-" * 72)

for name, predictions, nn_count in methods:
    accuracy = accuracy_score(y_test, predictions) * 100
    usage = nn_count / len(y_test) * 100

    print(
        f"{name:<25}{accuracy:>11.2f}%"
        f"{nn_count:>12}{usage:>12.2f}%"
    )

print("-" * 72)
print("Evaluation complete. No saved models or CSV files were modified.")
