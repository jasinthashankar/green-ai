
import time
import joblib

from sklearn.metrics import accuracy_score
from dataset import get_dataset
from cascading import predict_with_cascade
from routing import predict_with_router


# Load trained models
tree = joblib.load("models/decision_tree.pkl")
neural_net = joblib.load("models/neural_network.pkl")
router = joblib.load("models/router.pkl")

# Load the same test set for every method
_, _, X_test, _, _, y_test = get_dataset()


def evaluate(name, predict_function):
    start = time.perf_counter()
    predictions = predict_function()
    elapsed = time.perf_counter() - start

    accuracy = accuracy_score(y_test, predictions)

    print(
        f"{name:<20} "
        f"Accuracy: {accuracy * 100:6.2f}% | "
        f"Time: {elapsed:.6f} sec | "
        f"Average: {elapsed / len(X_test) * 1000:.4f} ms"
    )


print("\n--- Green AI Model Comparison ---")
print(f"Test samples: {len(X_test)}\n")

evaluate(
    "Decision Tree",
    lambda: tree.predict(X_test)
)

evaluate(
    "Neural Network",
    lambda: neural_net.predict(X_test)
)

evaluate(
    "Dynamic Cascading",
    lambda: predict_with_cascade(
        tree, neural_net, X_test,
        confidence_threshold=0.80
    )[0]
)

evaluate(
    "Dynamic Routing",
    lambda: predict_with_router(
        tree, neural_net, router, X_test
    )[0]
)
