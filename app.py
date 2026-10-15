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
from src.adaptive_selector import CostAwareAdaptiveSelector
from src.sla_camas import SLACAMASSelector


st.set_page_config(
    page_title="Green AI | Dynamic Model Selection",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Green AI: Dynamic Model Selection")
st.caption(
    "Multi-objective evaluation of computational efficiency and predictive accuracy across dynamic model selection architectures. "
    "Based on Cruciani & Verdecchia (ECAI 2025) and Peykani et al. (Springer 2026, DOI: 10.1007/s11831-026-10546-2)."
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
def load_all_data():
    X_train, X_val, X_test, y_train, y_val, y_test = get_dataset()
    return X_train, X_val, X_test, y_train, y_val, y_test


try:
    tree, neural_net, router = load_models()
    X_train, X_val, X_test, y_train, y_val, y_test = load_all_data()
except Exception as exc:
    st.error(
        "Could not load the dataset or models. "
        "Check that models/*.pkl exist and src/dataset.py scales pixels by 1/16."
    )
    st.exception(exc)
    st.stop()


# --------------------------------------------------
# Sidebar Settings
# --------------------------------------------------
st.sidebar.header("Experiment Settings")

cascade_threshold = st.sidebar.slider(
    "Cascade confidence threshold",
    min_value=0.50, max_value=0.99, value=0.80, step=0.05,
    help="Samples where tree confidence < threshold are escalated to the Neural Network."
)

sla_budget = st.sidebar.slider(
    "SLA-CAMAS NN invocation budget",
    min_value=0.00, max_value=1.00, value=0.50, step=0.05,
    help="Maximum fraction of samples permitted to escalate to the Neural Network. Calibrated on validation data."
)

repeats = st.sidebar.selectbox(
    "Timing repetitions",
    options=[30, 50, 100],
    index=1,
    help="Number of alternating benchmark repetitions used to measure median latency."
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Green AI Metric Note:**\n\n"
    "• **FLOPs:** Exact theoretical algorithmic complexity.\n"
    "• **Latency:** Wall-clock CPU time per batch.\n"
    "• **Energy/Carbon:** On Windows, CodeCarbon relies on constant-TDP heuristics. "
    "FLOPs and latency provide the most rigorous reproducible indicators."
)


# --------------------------------------------------
# Initialize CAMAS & SLA-CAMAS Selectors
# --------------------------------------------------
@st.cache_resource
def get_adaptive_selectors(_tree, _neural_net, budget_val):
    # Calibrated strictly on validation split
    camas = CostAwareAdaptiveSelector(_tree, _neural_net)
    camas.fit_policy(X_val, y_val)

    sla = SLACAMASSelector(_tree, _neural_net)
    sla.fit_budget_policy(X_val, y_val, max_nn_budget=budget_val)
    return camas, sla

camas_selector, sla_selector = get_adaptive_selectors(tree, neural_net, float(sla_budget))


# --------------------------------------------------
# The six methods (all batched, fair comparison)
# --------------------------------------------------
def run_tree(X):
    return tree.predict(X), 0


def run_nn(X):
    return neural_net.predict(X), len(X)


def run_cascade(X, threshold):
    proba = tree.predict_proba(X)
    pred = tree.classes_[proba.argmax(axis=1)].copy()
    hard = proba.max(axis=1) < threshold
    if hard.any():
        pred[hard] = neural_net.predict(X[hard])
    return pred, int(hard.sum())


def run_routing(X):
    proba = tree.predict_proba(X)
    pred = tree.classes_[proba.argmax(axis=1)].copy()
    use_nn = router.predict(proba) == 1
    if use_nn.any():
        pred[use_nn] = neural_net.predict(X[use_nn])
    return pred, int(use_nn.sum())


def run_camas(X):
    preds, nn_count, _ = camas_selector.predict(X)
    return preds, int(nn_count)


def run_sla(X):
    preds, nn_count, _ = sla_selector.predict(X)
    return preds, int(nn_count)


@st.cache_data(show_spinner=False)
def benchmark_all(c_thresh, s_budget, n_repeats):
    """Warm-up, alternate order every repeat, report median time."""
    methods = {
        "Decision Tree": lambda: run_tree(X_test),
        "Neural Network": lambda: run_nn(X_test),
        "Dynamic Cascade": lambda: run_cascade(X_test, c_thresh),
        "Dynamic Routing": lambda: run_routing(X_test),
        "CAMAS": lambda: run_camas(X_test),
        f"SLA-CAMAS @ {int(s_budget * 100)}%": lambda: run_sla(X_test),
    }

    # Warm-up passes
    for fn in methods.values():
        for _ in range(5):
            fn()

    times = {name: [] for name in methods}
    names = list(methods.keys())
    for i in range(n_repeats):
        order = names if i % 2 == 0 else names[::-1]
        for name in order:
            t0 = time.perf_counter()
            methods[name]()
            times[name].append((time.perf_counter() - t0) * 1000)

    # Theoretical FLOPs constants for Digits architecture
    flops_tree = 5.0
    flops_nn = 120970.0
    flops_overhead_routing = 130.0
    flops_overhead_camas = 75.0

    rows = []
    n_test = len(X_test)
    for name, fn in methods.items():
        pred, nn_count = fn()
        nn_pct = (nn_count / n_test) * 100

        # Calculate theoretical FLOPs per sample
        if name == "Decision Tree":
            flops = flops_tree
        elif name == "Neural Network":
            flops = flops_nn
        elif name == "Dynamic Cascade":
            flops = flops_tree + (nn_count / n_test) * flops_nn
        elif name == "Dynamic Routing":
            flops = flops_overhead_routing + (nn_count / n_test) * flops_nn
        else:  # CAMAS and SLA-CAMAS
            flops = flops_overhead_camas + (nn_count / n_test) * flops_nn

        rows.append({
            "Method": name,
            "Accuracy (%)": accuracy_score(y_test, pred) * 100,
            "NN usage (%)": nn_pct,
            "Median time (ms)": statistics.median(times[name]),
            "FLOPs/Sample": flops,
        })

    df = pd.DataFrame(rows)
    nn_row = df[df["Method"] == "Neural Network"].iloc[0]
    df["Time saved vs NN (%)"] = (1 - df["Median time (ms)"] / nn_row["Median time (ms)"]) * 100
    df["FLOPs saved vs NN (%)"] = (1 - df["FLOPs/Sample"] / nn_row["FLOPs/Sample"]) * 100
    df["Accuracy kept (%)"] = (df["Accuracy (%)"] / nn_row["Accuracy (%)"]) * 100
    return df


def read_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None


# --------------------------------------------------
# Navigation Tabs
# --------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Live Comparison",
    "Model-Size Study",
    "Threshold Trade-Off",
    "Try a Digit",
    "Pareto Frontier Analysis",
    "Wine Generalization",
])


# --------------------------------------------------
# Tab 1: Live Comparison
# --------------------------------------------------
with tab1:
    st.subheader("Live Multi-Method Benchmark (Digits Test Set)")
    st.write(
        f"Comparing all 6 selection strategies on {len(X_test)} test samples. "
        f"SLA-CAMAS threshold calibrated to honor **{int(sla_budget * 100)}% NN budget** "
        f"(calibrated threshold τ* = `{sla_selector.threshold_:.4f}`)."
    )

    with st.spinner("Benchmarking all six methods..."):
        results = benchmark_all(float(cascade_threshold), float(sla_budget), int(repeats))

    # Metric summary pills
    cols = st.columns(6)
    for col, name in zip(cols, results["Method"]):
        row = results[results["Method"] == name].iloc[0]
        sub = None
        if name not in ("Decision Tree", "Neural Network"):
            sub = f"{row['Time saved vs NN (%)']:.1f}% time saved"
        col.metric(name, f"{row['Accuracy (%)']:.2f}%", sub)

    st.dataframe(
        results[[
            "Method", "Accuracy (%)", "NN usage (%)",
            "Median time (ms)", "FLOPs/Sample",
            "Time saved vs NN (%)", "FLOPs saved vs NN (%)"
        ]].round(2),
        use_container_width=True,
        hide_index=True
    )

    left, right = st.columns(2)
    with left:
        st.write("**Accuracy (%) vs Methods**")
        st.bar_chart(results.set_index("Method")[["Accuracy (%)"]])
    with right:
        st.write("**FLOPs Saved vs Neural Network (%)**")
        st.bar_chart(results.set_index("Method")[["FLOPs saved vs NN (%)"]])

    st.caption(
        f"Evaluated on {len(X_test)} test samples. Median latencies computed over {repeats} alternating repetitions. "
        "Theoretical FLOPs represent architectural floating-point operations independent of hardware variations."
    )


# --------------------------------------------------
# Tab 2: Model-Size Study
# --------------------------------------------------
with tab2:
    st.header("When does dynamic model selection pay off?")
    st.write(
        "The Decision Tree is fixed and the Neural Network grows. "
        "Small networks are cheap enough that the selection overhead cancels the "
        "savings; large networks give significant computational and energy savings."
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
# Tab 3: Threshold Trade-Off
# --------------------------------------------------
with tab3:
    st.header("Cascade Confidence Threshold Trade-Off")
    st.write(
        "A higher confidence threshold sends more samples to the Neural Network: "
        "higher accuracy, but lower computational savings."
    )
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
        st.caption(
            "Thresholds are compared on validation data; test numbers are for final reporting. "
            "Equal NN usage across nearby thresholds happens because the tree's leaf probabilities take discrete values."
        )


# --------------------------------------------------
# Tab 4: Single Digit Prediction Demo
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
    features = np.asarray(preview, dtype=np.float64) / 255.0
    return image, preview, features.reshape(1, 64)


with tab4:
    st.header("Interactive Digit Prediction Demo")

    st.subheader("A. Pick a sample from the test set")
    idx = int(st.number_input(
        "Test sample number",
        min_value=0, max_value=len(X_test) - 1, value=0, step=1
    ))
    sample = X_test[idx].reshape(1, -1)
    true_label = int(y_test[idx])
    sample_img = Image.fromarray((sample.reshape(8, 8) * 255).astype(np.uint8))
    st.image(
        sample_img.resize((180, 180), Image.Resampling.NEAREST),
        caption=f"Ground Truth Digit: {true_label}"
    )

    sp_tree, _ = run_tree(sample)
    sp_nn, _ = run_nn(sample)
    sp_cas, sp_cas_used = run_cascade(sample, cascade_threshold)
    sp_rou, sp_rou_used = run_routing(sample)
    sp_cam, sp_cam_used = run_camas(sample)
    sp_sla, sp_sla_used = run_sla(sample)

    sample_preds = [
        int(sp_tree[0]), int(sp_nn[0]), int(sp_cas[0]),
        int(sp_rou[0]), int(sp_cam[0]), int(sp_sla[0])
    ]
    st.dataframe(
        pd.DataFrame({
            "Strategy": [
                "Decision Tree", "Neural Network", "Dynamic Cascade",
                "Dynamic Routing", "CAMAS", f"SLA-CAMAS @ {int(sla_budget * 100)}%"
            ],
            "Predicted Digit": sample_preds,
            "Correct?": ["Yes" if p == true_label else "No" for p in sample_preds],
            "Neural Network Used?": [
                "No", "Yes",
                "Yes" if sp_cas_used else "No",
                "Yes" if sp_rou_used else "No",
                "Yes" if sp_cam_used else "No",
                "Yes" if sp_sla_used else "No"
            ],
        }),
        use_container_width=True,
        hide_index=True
    )

    st.divider()
    st.subheader("B. Upload your own handwritten digit image")
    st.write("Upload an image with one digit (0-9) to see how all 6 strategies respond.")
    uploaded = st.file_uploader(
        "Choose a digit image",
        type=["png", "jpg", "jpeg"],
        key="digit_image_uploader"
    )
    if uploaded is not None:
        try:
            original, preview, feats = preprocess_uploaded_digit(uploaded)
            i1, i2 = st.columns(2)
            with i1:
                st.write("**Uploaded Image**")
                st.image(original, use_container_width=True)
            with i2:
                st.write("**Downsampled (8x8) Input**")
                st.image(preview.resize((200, 200), Image.Resampling.NEAREST), use_container_width=True)

            tp, _ = run_tree(feats)
            npred, _ = run_nn(feats)
            cp, c_used = run_cascade(feats, cascade_threshold)
            rp, r_used = run_routing(feats)
            camp, cam_used = run_camas(feats)
            slap, sla_used = run_sla(feats)

            st.subheader("Prediction Results")
            st.dataframe(
                pd.DataFrame({
                    "Strategy": [
                        "Decision Tree", "Neural Network", "Dynamic Cascade",
                        "Dynamic Routing", "CAMAS", f"SLA-CAMAS @ {int(sla_budget * 100)}%"
                    ],
                    "Predicted Digit": [
                        int(tp[0]), int(npred[0]), int(cp[0]),
                        int(rp[0]), int(camp[0]), int(slap[0])
                    ],
                    "Neural Network Used?": [
                        "No", "Yes",
                        "Yes" if c_used else "No",
                        "Yes" if r_used else "No",
                        "Yes" if cam_used else "No",
                        "Yes" if sla_used else "No"
                    ],
                }),
                use_container_width=True,
                hide_index=True
            )
        except Exception as exc:
            st.error(f"Could not process image: {exc}")


# --------------------------------------------------
# Tab 5: Pareto Frontier Analysis (Phase 4)
# --------------------------------------------------
with tab5:
    st.header("Phase 4: Multi-Objective Pareto Frontier Analysis")
    st.markdown(
        "In Green AI, optimization is inherently multi-objective: we seek to **maximize predictive accuracy** "
        "while **minimizing inference latency and computational FLOPs**. "
        "A configuration dominates another if it is no worse in all objectives and strictly superior in at least one."
    )

    p_digits_df = read_csv("results/pareto_digits.csv")
    p_wine_df = read_csv("results/pareto_wine.csv")

    p_tab1, p_tab2 = st.tabs(["Digits Dataset (Vision)", "Wine Dataset (Chemical)"])

    with p_tab1:
        st.subheader("Digits Multi-Objective Pareto Analysis")
        if p_digits_df is not None:
            show_cols = [
                "Method", "SLA Budget (%)", "Accuracy (%)", "NN Usage (%) [Test]",
                "Latency Median (ms)", "Theoretical FLOPs/Sample",
                "Pareto Optimal (Acc vs Latency)", "Pareto Optimal (Acc vs FLOPs)",
                "Pareto Optimal (3D: Acc, Latency, FLOPs)", "Dominated By (3D)"
            ]
            st.dataframe(p_digits_df[show_cols].round(3), use_container_width=True, hide_index=True)

            d_f1, d_f2 = st.columns(2)
            with d_f1:
                st.write("**Accuracy vs. Latency Pareto Frontier (Digits)**")
                if os.path.exists("results/figures/digits_accuracy_vs_latency.png"):
                    st.image("results/figures/digits_accuracy_vs_latency.png", use_container_width=True)
            with d_f2:
                st.write("**Accuracy vs. Theoretical FLOPs Frontier (Digits)**")
                if os.path.exists("results/figures/digits_accuracy_vs_flops.png"):
                    st.image("results/figures/digits_accuracy_vs_flops.png", use_container_width=True)
        else:
            st.warning("`results/pareto_digits.csv` not found. Run `python src/pareto_analysis.py`.")

    with p_tab2:
        st.subheader("Wine Multi-Objective Pareto Analysis")
        if p_wine_df is not None:
            show_cols = [
                "Method", "SLA Budget (%)", "Accuracy (%)", "NN Usage (%) [Test]",
                "Latency Median (ms)", "Theoretical FLOPs/Sample",
                "Pareto Optimal (Acc vs Latency)", "Pareto Optimal (Acc vs FLOPs)",
                "Pareto Optimal (3D: Acc, Latency, FLOPs)", "Dominated By (3D)"
            ]
            st.dataframe(p_wine_df[show_cols].round(3), use_container_width=True, hide_index=True)

            w_f1, w_f2 = st.columns(2)
            with w_f1:
                st.write("**Accuracy vs. Latency Pareto Frontier (Wine)**")
                if os.path.exists("results/figures/wine_accuracy_vs_latency.png"):
                    st.image("results/figures/wine_accuracy_vs_latency.png", use_container_width=True)
            with w_f2:
                st.write("**Accuracy vs. Theoretical FLOPs Frontier (Wine)**")
                if os.path.exists("results/figures/wine_accuracy_vs_flops.png"):
                    st.image("results/figures/wine_accuracy_vs_flops.png", use_container_width=True)
        else:
            st.warning("`results/pareto_wine.csv` not found. Run `python src/pareto_analysis.py`.")

    st.markdown("---")
    st.subheader("Dominance Findings & Methodological Rigor")
    st.markdown(
        """
        - **Digits Finding:** Decision Tree (lowest cost: 5 FLOPs), Neural Network (highest accuracy: 96.94%), Dynamic Cascade, Dynamic Routing, and intermediate SLA-CAMAS budgets establish the non-dominated Pareto frontier.
        - **3D Dominance nuance:** SLA-CAMAS @ 50% strictly dominates CAMAS because both achieve identical test accuracy (93.61%) and invoke the NN on 169 samples (56,443 FLOPs), while SLA-CAMAS displays slightly lower median latency (0.0766 ms vs 0.0861 ms).
        - **Wine Dominance nuance:** SLA-CAMAS @ 100% is dominated by the standalone Neural Network because routing overhead adds 17 FLOPs (5,969 vs 5,952) and extra dispatch latency without improving accuracy.
        - **Scientific Measurement Caveat:** On Windows OS environments without RAPL/NVML hardware counter drivers, CodeCarbon operates using static TDP estimates. Consequently, **theoretical FLOPs and high-precision median latency** serve as the definitive benchmark criteria.
        """
    )


# --------------------------------------------------
# Tab 6: Wine Generalization Study
# --------------------------------------------------
with tab6:
    st.header("Cross-Domain Validation: UCI Wine Dataset")
    st.markdown(
        "To test whether Green AI dynamic selection generalizes beyond pixel inputs, "
        "we validated the architecture on the tabular UCI Wine dataset (13 chemical attributes, 3 classes)."
    )

    w_bench_df = read_csv("results/wine_benchmark.csv")
    w_sla_df = read_csv("results/sla_camas_wine.csv")

    col_w1, col_w2 = st.columns(2)
    with col_w1:
        st.write("**Baseline 5-Method Benchmark (Wine)**")
        if w_bench_df is not None:
            st.dataframe(w_bench_df, use_container_width=True, hide_index=True)
        else:
            st.info("Run `python src/wine_experiment.py` to generate.")

    with col_w2:
        st.write("**SLA-CAMAS Budget Sweep (Wine)**")
        if w_sla_df is not None:
            st.dataframe(w_sla_df[[
                "Method", "SLA Budget (%)", "Accuracy (%)",
                "Val NN Usage (%)", "NN Usage (%) [Test]",
                "Latency Median (ms)", "Theoretical FLOPs Saved (%)"
            ]].round(2), use_container_width=True, hide_index=True)
        else:
            st.info("Run `python src/sla_benchmark.py` to generate.")

    st.info(
        "**Discrete Uncertainty Effect:** The Decision Tree on Wine has depth 2 with only 3 leaf nodes. "
        "As a result, leaf uncertainty scores take only 3 discrete values, producing step-function threshold behavior. "
        "This proves that dynamic selection requires trees with sufficient granularity to achieve continuous budget tuning."
    )