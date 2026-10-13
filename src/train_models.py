
import os
import time
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from dataset import get_dataset


# Create the models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Load dataset
X_train, X_val, X_test, y_train, y_val, y_test = get_dataset()

# 1. Train lightweight Decision Tree
print("Training Decision Tree...")

tree = DecisionTreeClassifier(max_depth=5, random_state=42)

start = time.perf_counter()
tree.fit(X_train, y_train)
tree_train_time = time.perf_counter() - start

tree_predictions = tree.predict(X_test)
tree_accuracy = accuracy_score(y_test, tree_predictions)

# 2. Train Neural Network
print("Training Neural Network...")

neural_net = MLPClassifier(
    hidden_layer_sizes=(64, 32, 16, 8, 4),
    max_iter=2000,
    random_state=42,
    early_stopping=False
)

start = time.perf_counter()
neural_net.fit(X_train, y_train)
nn_train_time = time.perf_counter() - start

nn_predictions = neural_net.predict(X_test)
nn_accuracy = accuracy_score(y_test, nn_predictions)

# Save trained models
joblib.dump(tree, "models/decision_tree.pkl")
joblib.dump(neural_net, "models/neural_network.pkl")

# Display results
print("\n========== MODEL RESULTS ==========")
print(f"Decision Tree accuracy: {tree_accuracy:.2%}")
print(f"Decision Tree training time: {tree_train_time:.4f} seconds")

print(f"\nNeural Network accuracy: {nn_accuracy:.2%}")
print(f"Neural Network training time: {nn_train_time:.4f} seconds")

print("\nModels saved successfully in the models folder!")
