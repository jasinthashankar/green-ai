
import time
import joblib
import numpy as np

from sklearn.metrics import accuracy_score
from dataset import get_dataset


def predict_with_router(tree, neural_net, router, X):
    predictions = []
    tree_count = 0
    nn_count = 0

    start = time.perf_counter()

    # First run only the Decision Tree
    tree_probabilities = tree.predict_proba(X)
    tree_predictions = tree.predict(X)

    # Decide which samples need the Neural Network
    choices = router.predict(tree_probabilities)
    predictions = tree_predictions.copy()

    nn_indices = np.where(choices == 1)[0]

    # Run the Neural Network only for selected samples
    if len(nn_indices) > 0:
        predictions[nn_indices] = neural_net.predict(X[nn_indices])

    nn_count = len(nn_indices)
    tree_count = len(X) - nn_count

    elapsed = time.perf_counter() - start

    return np.array(predictions), tree_count, nn_count, elapsed


if __name__ == "__main__":
    tree = joblib.load("models/decision_tree.pkl")
    neural_net = joblib.load("models/neural_network.pkl")
    router = joblib.load("models/router.pkl")

    _, _, X_test, _, _, y_test = get_dataset()

    predictions, tree_count, nn_count, elapsed = predict_with_router(
        tree, neural_net, router, X_test
    )

    total = len(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("\n--- Improved Dynamic Routing Results ---")
    print(f"Test samples: {total}")
    print(f"Routing accuracy: {accuracy * 100:.2f}%")
    print(f"Samples handled by Decision Tree: {tree_count}")
    print(f"Samples sent to Neural Network: {nn_count}")
    print(f"Neural Network usage: {nn_count / total * 100:.2f}%")
    print(f"Total inference time: {elapsed:.6f} seconds")
    print(f"Average inference time: {elapsed / total * 1000:.4f} ms")
