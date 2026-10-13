
import time
import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from dataset import get_dataset


def predict_cascade_singlepass(tree, neural_net, X, threshold=0.80):
    start = time.perf_counter()

    probabilities = tree.predict_proba(X)
    confidence = np.max(probabilities, axis=1)

    # Match the decision tree's class ordering.
    tree_predictions = tree.classes_[np.argmax(probabilities, axis=1)]

    uncertain_indices = np.flatnonzero(confidence < threshold)
    predictions = tree_predictions.copy()

    if len(uncertain_indices) > 0:
        predictions[uncertain_indices] = neural_net.predict(
            X[uncertain_indices]
        )

    elapsed = time.perf_counter() - start
    return predictions, len(uncertain_indices), elapsed


if __name__ == "__main__":
    tree = joblib.load("models/decision_tree.pkl")
    neural_net = joblib.load("models/neural_network.pkl")

    _, _, X_test, _, _, y_test = get_dataset()
    threshold = 0.80

    # Warm-up runs before timing.
    for _ in range(5):
        predict_cascade_singlepass(tree, neural_net, X_test, threshold)

    predictions, escalated, elapsed = predict_cascade_singlepass(
        tree, neural_net, X_test, threshold
    )

    accuracy = accuracy_score(y_test, predictions)

    print("\nSINGLE-PASS CASCADE EXPERIMENT")
    print("-" * 40)
    print("Test samples:", len(X_test))
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("NN escalations:", escalated)
    print(f"NN usage: {escalated / len(X_test) * 100:.2f}%")
    print(f"Elapsed time: {elapsed * 1000:.4f} ms")
    print("Existing models and result files were not modified.")
