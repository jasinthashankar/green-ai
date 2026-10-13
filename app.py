import os
import time
import statistics

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from sklearn.metrics import accuracy_score
from src.dataset import get_dataset


st.set_page_config(page_title="Green AI | Dynamic Model Selection",
                   page_icon="🌱", layout="wide")

st.title("🌱 Green AI: Dynamic Model Selection")
st.caption(
    "Cascading and routing between a cheap Decision Tree and an accurate "
    "Neural Network. Based on Cruciani & Verdecchia (2025), "
    "'Choosing to Be Green: Advancing Green AI via Dynamic Model Selection'."
)


# --------------------------------------------------
# Load models and data
# --------------------------------------------------
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
    st.error("Could not load the dataset or models. "
             "Check that models/*.pkl exist and src/dataset.py scales pixels by 1/16.")
    st.exception(exc)
    st.stop()


# --------------------------------------------------
# The four methods (all batched, so the comparison is fair)
# --------------------------------------------------
def run_tree(X):
    return tree.predict(X), 0


def run_nn(X):
    return neural_net.predict(X), len(X)


def run_cascade(X, threshold):
    """Tree runs once on everything; only low-confidence samples go to the NN."""
    proba = tree.predict_proba(X)
    pred = tree.classes_[proba.argmax(axis=1)]
    hard = proba.max(axis=1) < threshold
    if hard.any():
        pred[hard] = neural_net.predict(X[hard])
    return pred, int(hard.sum())


def run_routing(X):
    """Router looks at the tree's output and decides who answers."""
    proba = tree.predict_proba(X)
    pred = tree.classes_[proba.argmax(axis=1)]
    use_nn = router.predict(proba) == 1
    if use_nn.any():
        pred[use_nn] = neural_net.predict(X[use_nn])
    return pred, int(use_nn.sum())


@st.cache_data(show_spinner=False)
def benchmark_all(threshold, repeats):
    """Warm-up, alternate the order every repeat, report the median."""
    methods = {
        "Decision Tree": lambda: run_tree(X_test),
        "Neural Network": lambda: run_nn(X_test),
        "Dynamic Cascade": lambda: run_cascade(X_test, threshold),
        "Dynamic Routing": lambda: run_routing(X_test),
    }
    for fn in methods.values():
        for _ in range(10):
            fn()

    times = {name: [] for name in methods}
    names = list(methods)
    for i in range(repeats):
        order = names if i % 2 == 0 else names[::-1]
        for name in order:
            start = time.perf_counter()
            methods[name]()
            times[name].append((time.perf_counter() - start) * 1000)

    rows = []
    for name, fn in methods.items():
        pred, nn_count = fn()
        rows.append({
            "Method": name,
            "Accuracy (%)": accuracy_score(y_test, pred) * 100,
            "NN usage (%)": nn_count / len(X_test) * 100,
            "Median time (ms)": statistics.median(times[name]),
        })
    df = pd.DataFrame(rows)
    nn_row = df[df["Method"] == "Neural Network"].iloc[0]
    df["Time saved vs NN (%)"] = (1 - df["Median time (ms)"] / nn_row["Median time (ms)"]) * 100
    df["Accuracy kept (%)"] = df["Accuracy (%)"] / nn_row["Accuracy (%)"] * 100
    return df


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("Experiment settings")
threshold = st.sidebar.slider("Cascade confidence threshold",
                              min_value=0.50, max_value=0.99, value=0.80, step=0.05)
repeats = st.sidebar.selectbox("Timing repetitions", options=[30, 50, 100], index=1)
st.sidebar.info(
    "Timing is the main metric. Estimated energy from tools such as CodeCarbon "
    "scales with CPU time, so it adds little independent information here."
)


def read_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None


tab1, tab2, tab3, tab4 = st.tabs([
    "Live comparison", "Model-size study", "Threshold trade-off", "Try a digit",
])


# --------------------------------------------------
# Tab 1: live comparison
# --------------------------------------------------
with tab1:
    with st.spinner("Benchmarking all four methods..."):
        results = benchmark_all(float(threshold), int(repeats))

    total = len(X_test)
    c1, c2, c3, c4 = st.columns(4)
    for col, name in zip((c1, c2, c3, c4), results["Method"]):
        row = results[results["Method"] == name].iloc[0]
        col.metric(name, f"{row['Accuracy (%)']:.2f}%",
                   None if name in ("Decision Tree", "Neural Network")
                   else f"{row['Time saved vs NN (%)']:.1f}% time saved vs NN")

    st.dataframe(results.round(3), use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        st.write("**Accuracy (%)**")
        st.bar_chart(results.set_index("Method")[["Accuracy (%)"]])
    with right:
        st.write("**Median inference time (ms)**")
        st.bar_chart(results.set_index("Method")[["Median time (ms)"]])

    st.caption(
        f"{total} test samples. All methods use batch predictions after warm-up; "
        "times are medians with the run order alternated to reduce bias. "
        "Results depend on your CPU and background load."
    )


# --------------------------------------------------
# Tab 2: model-size study
# --------------------------------------------------
with tab2:
    st.header("When does dynamic model selection pay off?")
    st.write(
        "The Decision Tree is fixed and the Neural Network grows. "
        "Small networks are cheap enough that the selection overhead cancels the "
        "savings; large networks give big savings."
    )
    size_df = read_csv("results/model_size_study.csv")
    if size_df is None:
        st.warning("Run `python src/model_size_study.py` first.")
    else:
        st.dataframe(size_df, use_container_width=True, hide_index=True)
        chart_df = size_df.copy()
        chart_df["Model"] = [f"{i + 1}. {name}" for i, name in enumerate(chart_df["NN size"])]
        st.line_chart(chart_df.set_index("Model")[
            ["Cascade time saved (%)", "Routing time saved (%)"]])
        if os.path.exists("results/model_size_study.png"):
            st.image("results/model_size_study.png")


# --------------------------------------------------
# Tab 3: threshold trade-off
# --------------------------------------------------
with tab3:
    st.header("Accuracy vs Neural Network usage")
    st.write("A higher threshold sends more samples to the Neural Network: "
             "higher accuracy, smaller savings.")
    thr_df = read_csv("results/threshold_tradeoff.csv")
    if thr_df is None:
        st.warning("Run `python src/threshold_tradeoff.py` first.")
    else:
        st.dataframe(thr_df, use_container_width=True, hide_index=True)
        a, b = st.columns(2)
        with a:
            st.write("**Threshold vs test accuracy (%)**")
            st.line_chart(thr_df.set_index("Threshold")[["Test accuracy (%)"]])
        with b:
            st.write("**Threshold vs time saved vs NN (%)**")
            st.line_chart(thr_df.set_index("Threshold")[["Time saved vs NN (%)"]])
        st.caption("Thresholds are compared on validation data; test numbers are "
                   "for final reporting. Equal NN usage across nearby thresholds "
                   "happens because the tree's leaf probabilities take few values.")


# --------------------------------------------------
# Tab 4: single digit prediction
# --------------------------------------------------
def preprocess_uploaded_digit(uploaded_file):
    """Convert one uploaded digit into 64 features scaled to 0-1 (same as training)."""
    image = Image.open(uploaded_file).convert("L")
    pixels = np.asarray(image).copy()

    border = np.concatenate([pixels[0, :], pixels[-1, :], pixels[:, 0], pixels[:, -1]])
    if np.median(border) > 127:
        pixels = 255 - pixels

    rows, cols = np.where(pixels > 30)
    if len(rows) == 0:
        raise ValueError("No visible digit detected. Try another image.")

    cropped = Image.fromarray(pixels).crop((
        int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1))

    width, height = cropped.size
    side = max(width, height) + 4
    square = Image.new("L", (side, side), 0)
    square.paste(cropped, ((side - width) // 2, (side - height) // 2))

    preview = square.resize((8, 8), Image.Resampling.LANCZOS)
    # Models were trained on pixels / 16 (range 0-1): scale 0-255 down to 0-1.
    features = np.asarray(preview, dtype=np.float64) / 255.0
    return image, preview, features.reshape(1, 64)


with tab4:
    st.header("Digit prediction demo")

    st.subheader("A. Pick a digit from the test set (reliable demo)")
    idx = int(st.number_input("Test sample number", min_value=0,
                              max_value=len(X_test) - 1, value=0, step=1))
    sample = X_test[idx].reshape(1, -1)
    true_label = int(y_test[idx])
    sample_img = Image.fromarray((sample.reshape(8, 8) * 255).astype(np.uint8))
    st.image(sample_img.resize((200, 200), Image.Resampling.NEAREST),
             caption=f"True digit: {true_label}")

    sp_tree, _ = run_tree(sample)
    sp_nn, _ = run_nn(sample)
    sp_cas, sp_cas_used = run_cascade(sample, threshold)
    sp_rou, sp_rou_used = run_routing(sample)
    sample_preds = [int(sp_tree[0]), int(sp_nn[0]), int(sp_cas[0]), int(sp_rou[0])]
    st.dataframe(pd.DataFrame({
        "Strategy": ["Decision Tree", "Neural Network", "Dynamic Cascade", "Dynamic Routing"],
        "Predicted digit": sample_preds,
        "Correct?": ["Yes" if p == true_label else "No" for p in sample_preds],
        "Neural Network used": ["No", "Yes",
                                "Yes" if sp_cas_used else "No",
                                "Yes" if sp_rou_used else "No"],
    }), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("B. Upload your own image (experimental)")
    st.write("Upload a clear image with one handwritten digit (0-9) and compare "
             "the four strategies on the same input.")
    uploaded = st.file_uploader("Choose a digit image", type=["png", "jpg", "jpeg"],
                                key="digit_image_uploader")
    if uploaded is not None:
        try:
            original, preview, feats = preprocess_uploaded_digit(uploaded)
            i1, i2 = st.columns(2)
            with i1:
                st.write("**Uploaded image**")
                st.image(original, use_container_width=True)
            with i2:
                st.write("**Processed 8 x 8 image**")
                st.image(preview.resize((240, 240), Image.Resampling.NEAREST),
                         use_container_width=True)

            tp, _ = run_tree(feats)
            npred, _ = run_nn(feats)
            cp, c_used = run_cascade(feats, threshold)
            rp, r_used = run_routing(feats)

            st.subheader("Prediction results")
            st.dataframe(pd.DataFrame({
                "Strategy": ["Decision Tree", "Neural Network",
                             "Dynamic Cascade", "Dynamic Routing"],
                "Predicted digit": [int(tp[0]), int(npred[0]), int(cp[0]), int(rp[0])],
                "Neural Network used": ["No", "Yes",
                                        "Yes" if c_used else "No",
                                        "Yes" if r_used else "No"],
            }), use_container_width=True, hide_index=True)
            st.info("These are predictions, not verified answers. "
                    "Photo preprocessing strongly affects accuracy on 8x8 digits.")
        except Exception as exc:
            st.error(f"Could not process this image: {exc}")