
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import accuracy_score
from src.dataset import get_dataset


st.set_page_config(
    page_title="Green AI | Model Evaluation",
    page_icon="🌱",
    layout="wide",
)

st.title("🌱 Green AI: Dynamic Model Selection")
st.caption(
    "Compare model accuracy, inference latency, and "
    "dynamic model selection strategies."
)

# Load all trained models
@st.cache_resource
def load_models():
    tree = joblib.load("models/decision_tree.pkl")
    neural_net = joblib.load("models/neural_network.pkl")
    router = joblib.load("models/router.pkl")
    return tree, neural_net, router


@st.cache_data
def load_test_data():
    _, _, X_test, _, _, y_test = get_dataset()
    return X_test, y_test


try:
    tree, neural_net, router = load_models()
    X_test, y_test = load_test_data()
except Exception as exc:
    st.error(
        "Could not load the dataset or models. "
        "Run src/train_models.py and src/train_router.py first."
    )
    st.exception(exc)
    st.stop()


st.sidebar.header("Experiment Settings")

threshold = st.sidebar.slider(
    "Decision Tree confidence threshold",
    min_value=0.50,
    max_value=0.99,
    value=0.80,
    step=0.05,
)

repeats = st.sidebar.selectbox(
    "Timing repetitions",
    options=[3, 5, 10],
    index=1,
)

st.sidebar.info(
    "Use validation data to select final settings. "
    "Keep the test set for final evaluation."
)


# Dynamic cascading
def run_cascade(X, confidence_threshold):
    predictions = []
    escalated = 0

    for sample in X:
        sample_2d = sample.reshape(1, -1)
        probabilities = tree.predict_proba(sample_2d)[0]

        if np.max(probabilities) >= confidence_threshold:
            prediction = tree.predict(sample_2d)[0]
        else:
            prediction = neural_net.predict(sample_2d)[0]
            escalated += 1

        predictions.append(prediction)

    return np.array(predictions), escalated


# Dynamic routing
def run_routing(X):
    # The router was trained using these tree probability features.
    tree_probabilities = tree.predict_proba(X)
    tree_predictions = tree.predict(X)

    choices = router.predict(tree_probabilities)
    predictions = tree_predictions.copy()

    nn_indices = np.where(choices == 1)[0]

    if len(nn_indices) > 0:
        predictions[nn_indices] = neural_net.predict(X[nn_indices])

    return predictions, len(nn_indices)


# Repeat each method and use median elapsed time
def benchmark(function, number_of_repeats):
    durations = []
    output = None

    for _ in range(number_of_repeats):
        start = time.perf_counter()
        output = function()
        durations.append(time.perf_counter() - start)

    return output, float(np.median(durations))


with st.spinner("Evaluating all four methods..."):
    tree_predictions, tree_seconds = benchmark(
        lambda: tree.predict(X_test), repeats
    )

    nn_predictions, nn_seconds = benchmark(
        lambda: neural_net.predict(X_test), repeats
    )

    (cascade_predictions, escalated), cascade_seconds = benchmark(
        lambda: run_cascade(X_test, threshold), repeats
    )

    (routing_predictions, routed_to_nn), routing_seconds = benchmark(
        lambda: run_routing(X_test), repeats
    )


# Calculate accuracy
tree_accuracy = accuracy_score(y_test, tree_predictions)
nn_accuracy = accuracy_score(y_test, nn_predictions)
cascade_accuracy = accuracy_score(y_test, cascade_predictions)
routing_accuracy = accuracy_score(y_test, routing_predictions)

total = len(X_test)

st.subheader("Experiment Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Test samples", total)
col2.metric("Decision Tree", f"{tree_accuracy:.2%}")
col3.metric("Neural Network", f"{nn_accuracy:.2%}")
col4.metric("Dynamic Cascade", f"{cascade_accuracy:.2%}")

st.metric("Dynamic Routing accuracy", f"{routing_accuracy:.2%}")

# Build comparison table
comparison = pd.DataFrame({
    "Method": [
        "Decision Tree",
        "Neural Network",
        "Dynamic Cascade",
        "Dynamic Routing",
    ],
    "Accuracy (%)": [
        tree_accuracy * 100,
        nn_accuracy * 100,
        cascade_accuracy * 100,
        routing_accuracy * 100,
    ],
    "Median inference time (ms)": [
        tree_seconds * 1000,
        nn_seconds * 1000,
        cascade_seconds * 1000,
        routing_seconds * 1000,
    ],
    "Neural Network usage (%)": [
        0,
        100,
        escalated / total * 100,
        routed_to_nn / total * 100,
    ],
})

st.subheader("Comparison of All Four Methods")

st.dataframe(
    comparison.round(3),
    use_container_width=True,
    hide_index=True,
)

left, right = st.columns(2)

with left:
    st.write("**Prediction accuracy (%)**")
    st.bar_chart(
        comparison.set_index("Method")[["Accuracy (%)"]]
    )

with right:
    st.write("**Median inference time (ms)**")
    st.bar_chart(
        comparison.set_index("Method")[["Median inference time (ms)"]]
    )

st.subheader("Model Selection Details")

col1, col2, col3 = st.columns(3)

col1.metric("Cascade threshold", f"{threshold:.0%}")
col2.metric(
    "Cascade NN escalations",
    f"{escalated} / {total}",
)
col3.metric(
    "Routing NN selections",
    f"{routed_to_nn} / {total}",
)

st.progress(min(escalated / total, 1.0))

st.caption(
    "Timing uses the median of repeated local runs. "
    "The cascade uses per-sample predictions, while the baseline "
    "models use batch predictions, so execution methods differ. "
    "Timing measures latency, not energy consumption."
)

st.warning(
    "These results are for experimentation. Do not claim energy savings "
    "without direct energy measurement and a fair comparison. "
    "Use validation data for tuning and the held-out test data "
    "for final reporting."
)
