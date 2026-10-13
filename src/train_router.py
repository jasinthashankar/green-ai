
import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from dataset import get_dataset


tree = joblib.load("models/decision_tree.pkl")
neural_net = joblib.load("models/neural_network.pkl")

# Use validation data only for training the router
_, X_val, _, _, y_val, _ = get_dataset()

# Use the tree's probability distribution as router input
tree_probabilities = tree.predict_proba(X_val)

tree_predictions = tree.predict(X_val)
nn_predictions = neural_net.predict(X_val)

# 1 means use Neural Network; 0 means use Decision Tree
router_labels = []

for i in range(len(y_val)):
    tree_correct = tree_predictions[i] == y_val[i]
    nn_correct = nn_predictions[i] == y_val[i]

    if not tree_correct and nn_correct:
        router_labels.append(1)
    else:
        # Prefer the simpler model if it is correct,
        # or if neither model can give the correct answer
        router_labels.append(0)

# Train the router using only tree probabilities
router = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

router.fit(tree_probabilities, router_labels)

os.makedirs("models", exist_ok=True)
joblib.dump(router, "models/router.pkl")

print("Improved router trained successfully!")
print(f"Validation samples: {len(X_val)}")
print(
    "Router training accuracy: "
    f"{accuracy_score(router_labels, router.predict(tree_probabilities)) * 100:.2f}%"
)
print(f"Tree choices in training labels: {router_labels.count(0)}")
print(f"Neural Network choices in training labels: {router_labels.count(1)}")
print("Saved to models/router.pkl")
