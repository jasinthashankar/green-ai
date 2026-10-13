
import time
import joblib
import numpy as np

from sklearn.metrics import accuracy_score
from dataset import get_dataset


def predict_with_cascade(
    tree,
    neural_net,
    X,
    confidence_threshold=0.80
):
    predictions = []
    escalated = 0

    start = time.perf_counter()

    for sample in X:
        sample = sample.reshape(1, -1)

        # Get class probabilities from the lightweight model
        probabilities = tree.predict_proba(sample)[0]
        confidence = np.max(probabilities)

        if confidence >= confidence_threshold:
            # Accept the Decision Tree prediction
            prediction = tree.predict(sample)[0]
        else:
            # Use the more complex model for uncertain samples
            prediction = neural_net.predict(sample)[0]
            escalated += 1

        predictions.append(prediction)

    elapsed = time.perf_counter() - start

    return (
        np.array(predictions),
        escalated,
        elapsed
    )


if __name__ == "__main__":
    # Load saved models
    tree = joblib.load("models/decision_tree.pkl")
    neural_net = joblib.load("models/neural_network.pkl")

    # Load the same held-out test set
    _, _, X_test, _, _, y_test = get_dataset()

    threshold = 0.80

    predictions, escalated, elapsed = predict_with_cascade(
        tree,
        neural_net,
        X_test,
        confidence_threshold=threshold
    )

    accuracy = accuracy_score(y_test, predictions)
    escalation_rate = escalated / len(X_test) * 100
    average_latency = elapsed / len(X_test) * 1000

    print("\n========== CASCADING RESULTS ==========")
    print(f"Confidence threshold: {threshold:.0%}")
    print(f"Cascade accuracy: {accuracy:.2%}")
    print(f"Samples tested: {len(X_test)}")
    print(f"Samples sent to Neural Network: {escalated}")
    print(f"Escalation rate: {escalation_rate:.2f}%")
    print(f"Total inference time: {elapsed:.6f} seconds")
    print(f"Average inference time: {average_latency:.4f} ms")
