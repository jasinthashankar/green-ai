
import time
import joblib
import numpy as np

from sklearn.metrics import accuracy_score
from dataset import get_dataset


def predict_cascade_batch(tree, neural_net, X, threshold=0.80):
    """Run the tree in batches and escalate only uncertain samples."""

    start = time.perf_counter()

    # One batch call obtains both predictions and confidence.
    probabilities = tree.predict_proba(X)
    tree_predictions = tree.predict(X)

    confidence = np.max(probabilities, axis=1)
    uncertain_indices = np.flatnonzero(confidence < threshold)

    predictions = tree_predictions.copy()

    # Call the neural network only for uncertain samples.
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

    # Warm-up calls are excluded from the reported timing.
    for _ in range(5):
        predict_cascade_batch(tree, neural_net, X_test, threshold)

    predictions, escalated, elapsed = predict_cascade_batch(
        tree, neural_net, X_test, threshold
    )

    accuracy = accuracy_score(y_test, predictions)

    print("\nBATCH CASCADE EXPERIMENT")
    print("-" * 40)
    print("Test samples:", len(X_test))
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("NN escalations:", escalated)
    print(f"NN usage: {escalated / len(X_test) * 100:.2f}%")
    print(f"Elapsed time: {elapsed * 1000:.4f} ms")
    print("Existing models and result files were not modified.")
 