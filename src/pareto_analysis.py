"""
Multi-Objective Pareto Frontier Analysis for Green AI Dynamic Model Selection.

Motivated by multi-tier evaluation principles and trade-off analysis in:
'Green Artificial Intelligence: A Comprehensive Review of Metrics, Tools, Challenges,
Trends, and Future Prospects' (2026), Springer, DOI: 10.1007/s11831-026-10546-2.

This module provides a rigorous, deterministic Pareto frontier analysis across:
1. Decision Tree (cheap baseline)
2. Neural Network (high-capacity ceiling)
3. Dynamic Cascade (confidence-thresholded)
4. Dynamic Routing (logistic-regression router)
5. CAMAS (cost-aware multi-criteria adaptive selector)
6. SLA-CAMAS (budget-constrained adaptive selector across multiple NN invocation budgets)

Key Design Principles & Methodological Boundaries:
- Independent Frontiers: Digits and Wine are analyzed separately. They feature distinct
  dimensionalities (64 vs 13 features), class cardinalities (10 vs 3 classes), test set sizes
  (360 vs 36 samples), and model architectures (60k vs 6k parameters). Combining raw metrics
  into a single frontier would introduce aggregation bias.
- Multi-Objective Space:
  * Accuracy (%) [Maximize]
  * Inference Latency Median (ms) [Minimize]
  * Theoretical Computational Cost (FLOPs/Sample) [Minimize]
- Energy Profiling Transparency: Energy and emissions (measured via CodeCarbon) are documented
  where validly profiled (base 5 methods), flagged with measurement limitations (Windows constant-TDP
  heuristic), and marked as unprofiled for SLA budget sweeps. They are NOT used to artificially
  penalize or dominate configurations where energy measurements were not performed.
- No False Composite Metrics: Avoids composite Green AI single-score formulas or Energy-to-Accuracy
  ratios that conceal non-linear trade-offs or break down when accuracy differentials approach zero.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Constants & Measurement Documentation
# --------------------------------------------------------------------------

ENERGY_LIMITATIONS = {
    "software_tracker": "CodeCarbon OfflineEmissionsTracker (country ISO: IND)",
    "sensor_status": (
        "On Windows without direct hardware RAPL/NVML drivers, CodeCarbon employs "
        "a constant TDP / CPU package heuristic. Measurements scale proportionally with runtime."
    ),
    "micro_inference_aggregation": (
        "Individual test set evaluations take <3 ms, well below the 1,000 ms polling resolution "
        "of software emissions trackers. Energy is only available for experiments executed under "
        "synthetic multi-trial repeat loops (1,000 repeats x 3 trials for Digits; 2,000 repeats x 3 trials for Wine)."
    ),
    "flops_vs_energy_distinction": (
        "Theoretical FLOPs calculate algorithmic operations and do not directly equate to hardware Joules "
        "due to memory bandwidth, instruction pipelining, and cache hierarchy effects."
    ),
    "sla_runs_note": (
        "SLA-CAMAS budget sweeps evaluate operational latency and FLOPs reduction. Energy profiling "
        "was not executed for intermediate SLA runs to avoid repeated synthetic loop overhead; values "
        "are preserved as NaN rather than fabricated."
    ),
}


# --------------------------------------------------------------------------
# Core Deterministic Pareto Dominance Engine
# --------------------------------------------------------------------------

def is_dominated(
    candidate: Union[Dict[str, Any], pd.Series],
    others: Union[List[Union[Dict[str, Any], pd.Series]], pd.DataFrame],
    objectives: List[Tuple[str, str]],
    tolerance: float = 1e-9,
    nan_policy: str = "raise",
) -> Tuple[bool, List[str]]:
    """
    Evaluates whether a candidate configuration is Pareto-dominated by any other configuration.

    Definition of Pareto Dominance:
        Point A dominates Point B (A ≻ B) if and only if:
        1. A is at least as good as B on ALL objectives:
           - For maximized objectives: A_k >= B_k - tolerance
           - For minimized objectives: A_k <= B_k + tolerance
        2. A is strictly better than B on AT LEAST ONE objective:
           - For maximized objectives: A_k > B_k + tolerance
           - For minimized objectives: A_k < B_k - tolerance

    Ties and Duplicate Points:
        If Point A and Point B have identical values on all objectives within tolerance,
        neither strictly dominates the other. If no other point dominates them, both
        remain non-dominated (Pareto-optimal).

    Parameters
    ----------
    candidate : dict or pd.Series
        The point being evaluated for dominance.
    others : list of dicts/pd.Series or pd.DataFrame
        The reference set of competing points.
    objectives : list of tuples (col_name, direction)
        direction must be 'max' (maximize) or 'min' (minimize).
    tolerance : float, default=1e-9
        Numerical precision tolerance for floating-point comparisons.
    nan_policy : str, default='raise'
        How to handle NaN values in objective columns:
        - 'raise': Raises ValueError if NaN is encountered.
        - 'omit': Skips comparison with rows containing NaN.
        - 'incomparable': If candidate has NaN, marks as dominated=False, dominators=[].

    Returns
    -------
    dominated : bool
        True if at least one other point dominates the candidate.
    dominators : list of str
        Identifiers/names of the configurations that dominate the candidate.
    """
    for col, direction in objectives:
        if direction not in ("max", "min"):
            raise ValueError(f"Invalid objective direction '{direction}' for '{col}'. Use 'max' or 'min'.")

    # Check candidate for NaNs
    cand_has_nan = any(pd.isna(candidate.get(col, np.nan)) for col, _ in objectives)
    if cand_has_nan:
        if nan_policy == "raise":
            missing_cols = [col for col, _ in objectives if pd.isna(candidate.get(col, np.nan))]
            raise ValueError(f"Candidate contains NaN in objective(s): {missing_cols}")
        elif nan_policy == "incomparable":
            return False, []
        elif nan_policy == "omit":
            return False, []

    dominators: List[str] = []

    # Iterate over competitors
    competitor_rows = others.to_dict("records") if isinstance(others, pd.DataFrame) else others

    for other in competitor_rows:
        other_name = other.get("Method", other.get("name", "Unknown"))
        # Check if identical identity (skip self-comparison)
        if other is candidate:
            continue

        other_has_nan = any(pd.isna(other.get(col, np.nan)) for col, _ in objectives)
        if other_has_nan:
            if nan_policy == "raise":
                missing_cols = [col for col, _ in objectives if pd.isna(other.get(col, np.nan))]
                raise ValueError(f"Competitor '{other_name}' contains NaN in objective(s): {missing_cols}")
            elif nan_policy in ("omit", "incomparable"):
                continue

        at_least_as_good = True
        strictly_better = False

        for col, direction in objectives:
            val_cand = float(candidate[col])
            val_other = float(other[col])

            if direction == "max":
                if val_other < val_cand - tolerance:
                    at_least_as_good = False
                    break
                elif val_other > val_cand + tolerance:
                    strictly_better = True
            else:  # direction == 'min'
                if val_other > val_cand + tolerance:
                    at_least_as_good = False
                    break
                elif val_other < val_cand - tolerance:
                    strictly_better = True

        if at_least_as_good and strictly_better:
            dominators.append(str(other_name))

    return (len(dominators) > 0), dominators


def compute_pareto_frontier(
    df: pd.DataFrame,
    objectives: List[Tuple[str, str]],
    tolerance: float = 1e-9,
    nan_policy: str = "raise",
) -> Tuple[pd.Series, pd.Series]:
    """
    Computes the Pareto frontier for a DataFrame across specified objective columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input data table.
    objectives : list of (col_name, 'max' | 'min')
    tolerance : float, default=1e-9
    nan_policy : str, default='raise'

    Returns
    -------
    is_pareto : pd.Series of bool
        True if the row is non-dominated.
    dominated_by : pd.Series of str
        Comma-separated list of dominating method names (empty string if Pareto-optimal).
    """
    is_pareto_list = []
    dominated_by_list = []

    rows = df.to_dict("records")
    for i, candidate in enumerate(rows):
        other_rows = [r for j, r in enumerate(rows) if j != i]
        dominated, dominators = is_dominated(
            candidate=candidate,
            others=other_rows,
            objectives=objectives,
            tolerance=tolerance,
            nan_policy=nan_policy,
        )
        is_pareto_list.append(not dominated)
        dominated_by_list.append(", ".join(dominators) if dominators else "")

    return (
        pd.Series(is_pareto_list, index=df.index, name="is_pareto"),
        pd.Series(dominated_by_list, index=df.index, name="dominated_by"),
    )


# --------------------------------------------------------------------------
# Data Loading & Enrichment Pipeline
# --------------------------------------------------------------------------

def load_and_enrich_dataset_results(
    dataset_name: str,
    sla_csv_path: str,
    base_csv_path: str,
) -> pd.DataFrame:
    """
    Loads SLA benchmark results and joins profiled CodeCarbon energy metrics from base benchmarks.
    Preserves exact distinction between measured energy, theoretical FLOPs, and unprofiled cells.

    Parameters
    ----------
    dataset_name : str ('Digits' or 'Wine')
    sla_csv_path : str (e.g. 'results/sla_camas_digits.csv')
    base_csv_path : str (e.g. 'results/enhanced_benchmark.csv')

    Returns
    -------
    df : pd.DataFrame with standardized metadata and metrics.
    """
    if not os.path.exists(sla_csv_path):
        raise FileNotFoundError(f"SLA benchmark file not found: {sla_csv_path}")

    sla_df = pd.read_csv(sla_csv_path)

    # Load energy data from base benchmark if available
    energy_map: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(base_csv_path):
        base_df = pd.read_csv(base_csv_path)
        for _, r in base_df.iterrows():
            m = str(r["Method"])
            energy_map[m] = {
                "energy_kwh": r.get("CodeCarbon Energy (kWh)", np.nan),
                "emissions_gco2e": r.get("CodeCarbon Emissions (gCO2e)", np.nan),
                "sensor_status": r.get("Energy Sensor Status", "Unknown"),
            }

    repeats_note = "1,000 repeats x 3 trials" if dataset_name == "Digits" else "2,000 repeats x 3 trials"

    rows = []
    for _, row in sla_df.iterrows():
        m = str(row["Method"])
        r_dict = row.to_dict()

        if m in energy_map:
            en_info = energy_map[m]
            r_dict["CodeCarbon Energy (kWh)"] = en_info["energy_kwh"]
            r_dict["CodeCarbon Emissions (gCO2e)"] = en_info["emissions_gco2e"]
            r_dict["Energy Sensor Status"] = en_info["sensor_status"]
            r_dict["Energy Measurement Source"] = f"CodeCarbon OfflineEmissionsTracker ({repeats_note})"
            r_dict["Energy Limitation Note"] = ENERGY_LIMITATIONS["sensor_status"]
        else:
            r_dict["CodeCarbon Energy (kWh)"] = np.nan
            r_dict["CodeCarbon Emissions (gCO2e)"] = np.nan
            r_dict["Energy Sensor Status"] = "Not profiled in SLA runs"
            r_dict["Energy Measurement Source"] = "Unavailable for intermediate SLA sweep"
            r_dict["Energy Limitation Note"] = ENERGY_LIMITATIONS["sla_runs_note"]

        r_dict["Data Source"] = f"{sla_csv_path} + {base_csv_path}"
        r_dict["Experiment Configuration"] = (
            f"{dataset_name} test partition ({row['Test Samples']} samples), "
            f"150 latency repetitions with 20 warmup cycles"
        )
        rows.append(r_dict)

    enriched_df = pd.DataFrame(rows)
    return enriched_df


def analyze_dataset_pareto(
    dataset_name: str,
    sla_csv_path: str,
    base_csv_path: str,
    output_csv_path: str,
) -> pd.DataFrame:
    """
    Executes complete multi-objective Pareto analysis on a single dataset.
    Computes:
    - 2D Frontier: Accuracy vs Latency
    - 2D Frontier: Accuracy vs Theoretical FLOPs
    - 3D Frontier: Accuracy vs Latency vs Theoretical FLOPs
    """
    df = load_and_enrich_dataset_results(dataset_name, sla_csv_path, base_csv_path)

    # 1. 2D Accuracy vs Latency
    obj_acc_lat = [("Accuracy (%)", "max"), ("Latency Median (ms)", "min")]
    p_acc_lat, _ = compute_pareto_frontier(df, obj_acc_lat)

    # 2. 2D Accuracy vs FLOPs
    obj_acc_flops = [("Accuracy (%)", "max"), ("Theoretical FLOPs/Sample", "min")]
    p_acc_flops, _ = compute_pareto_frontier(df, obj_acc_flops)

    # 3. 3D Joint Objective (Accuracy, Latency, FLOPs)
    obj_3d = [
        ("Accuracy (%)", "max"),
        ("Latency Median (ms)", "min"),
        ("Theoretical FLOPs/Sample", "min"),
    ]
    p_3d, dom_by_3d = compute_pareto_frontier(df, obj_3d)

    # Identify ties/duplicates on 3D coordinates
    coord_counts = df.groupby(["Accuracy (%)", "Latency Median (ms)", "Theoretical FLOPs/Sample"]).size()
    is_tie = df.apply(
        lambda r: coord_counts.get((r["Accuracy (%)"], r["Latency Median (ms)"], r["Theoretical FLOPs/Sample"]), 1) > 1,
        axis=1,
    )

    df["Pareto Optimal (Acc vs Latency)"] = p_acc_lat
    df["Pareto Optimal (Acc vs FLOPs)"] = p_acc_flops
    df["Pareto Optimal (3D: Acc, Latency, FLOPs)"] = p_3d
    df["Dominated By (3D)"] = dom_by_3d
    df["Is Tie or Duplicate"] = is_tie

    # Write output CSV
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    return df


# --------------------------------------------------------------------------
# Publication-Quality Visualization Engine
# --------------------------------------------------------------------------

def plot_pareto_frontier_2d(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    output_path: str,
    pareto_col: str,
    dataset_name: str,
    use_log_x: bool = False,
) -> None:
    """
    Generates high-resolution publication-quality Pareto frontier scatter plots.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    # Color palette and marker configurations
    color_map = {
        "Decision Tree": "#2ca02c",             # Forest green
        "Neural Network": "#7b1fa2",            # Deep purple
        "Dynamic Cascade": "#e65100",           # Vivid amber/orange
        "Dynamic Routing": "#0097a7",           # Teal/Cyan
        "CAMAS (Adaptive Selection)": "#1565c0",# Royal blue
    }
    default_sla_color = "#d81b60"               # Magenta/Pink for SLA-CAMAS

    # Separate Pareto-optimal from dominated
    pareto_mask = df[pareto_col].astype(bool)
    pareto_df = df[pareto_mask].sort_values(by=x_col)
    dominated_df = df[~pareto_mask]

    # Draw Pareto frontier line (step/curve connecting non-dominated points)
    if len(pareto_df) > 1:
        ax.plot(
            pareto_df[x_col],
            pareto_df[y_col],
            linestyle="--",
            color="#d32f2f",
            linewidth=1.8,
            alpha=0.75,
            label="Pareto Frontier (Non-dominated)",
            zorder=2,
        )

    # Plot dominated points
    for _, row in dominated_df.iterrows():
        m = str(row["Method"])
        c = color_map.get(m, default_sla_color)
        ax.scatter(
            row[x_col],
            row[y_col],
            color=c,
            alpha=0.45,
            edgecolors="gray",
            linewidths=1.0,
            s=85,
            marker="o",
            zorder=3,
        )

    # Plot Pareto-optimal points
    for _, row in pareto_df.iterrows():
        m = str(row["Method"])
        c = color_map.get(m, default_sla_color)
        ax.scatter(
            row[x_col],
            row[y_col],
            color=c,
            alpha=0.95,
            edgecolors="#1a237e",
            linewidths=1.8,
            s=140,
            marker="D" if "SLA" in m else "s",
            zorder=4,
        )

    # Annotations with smart offsets
    for _, row in df.iterrows():
        m = str(row["Method"])
        # Format label cleanly
        if m.startswith("SLA-CAMAS @ "):
            label = m.replace("SLA-CAMAS @ ", "SLA ")
        elif m == "CAMAS (Adaptive Selection)":
            label = "CAMAS"
        else:
            label = m

        x_val = row[x_col]
        y_val = row[y_col]
        is_opt = bool(row[pareto_col])

        # Stagger annotations based on method to prevent overlaps
        dx, dy = 6, 4
        if "Tree" in m:
            dx, dy = 8, -8
        elif "Neural" in m:
            dx, dy = -60, -12
        elif "Cascade" in m:
            dx, dy = -55, 6
        elif "Routing" in m:
            dx, dy = 8, -12
        elif "CAMAS" in m:
            dx, dy = 8, 4

        weight = "bold" if is_opt else "normal"
        color = "#111111" if is_opt else "#555555"

        ax.annotate(
            label,
            xy=(x_val, y_val),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.5,
            fontweight=weight,
            color=color,
            zorder=5,
        )

    if use_log_x:
        ax.set_xscale("log")

    ax.set_xlabel(x_label, fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel(y_label, fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    # Clean styling
    ax.grid(True, linestyle=":", alpha=0.6, zorder=1)
    ax.set_axisbelow(True)

    # Custom legend items
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], linestyle="--", color="#d32f2f", lw=1.8, label="Pareto Frontier"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#d81b60", markeredgecolor="#1a237e",
               markersize=9, label="Pareto-Optimal Point"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markeredgecolor="gray",
               alpha=0.5, markersize=7, label="Dominated Point"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------
# Summary Report Generator
# --------------------------------------------------------------------------

def generate_summary_markdown(
    digits_df: pd.DataFrame,
    wine_df: pd.DataFrame,
    output_md_path: str,
) -> None:
    """
    Generates a comprehensive research summary report of Phase 4 Pareto Analysis.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_md_path)), exist_ok=True)

    def _format_table(df: pd.DataFrame) -> str:
        cols = [
            "Method", "SLA Budget (%)", "Accuracy (%)", "NN Usage (%) [Test]",
            "Latency Median (ms)", "Theoretical FLOPs/Sample",
            "Pareto Optimal (Acc vs Latency)", "Pareto Optimal (Acc vs FLOPs)",
            "Pareto Optimal (3D: Acc, Latency, FLOPs)", "Dominated By (3D)"
        ]
        sub = df[cols].copy()
        # Convert booleans to clean checkmarks
        for b_col in ["Pareto Optimal (Acc vs Latency)", "Pareto Optimal (Acc vs FLOPs)", "Pareto Optimal (3D: Acc, Latency, FLOPs)"]:
            sub[b_col] = sub[b_col].apply(lambda v: "✓ Yes" if v else "No")
        sub["SLA Budget (%)"] = sub["SLA Budget (%)"].fillna("—")
        sub["Dominated By (3D)"] = sub["Dominated By (3D)"].apply(lambda v: v if str(v) != "" and str(v) != "nan" else "—")
        
        # Pure Python markdown table formatter (no tabulate dependency)
        headers = list(sub.columns)
        lines = []
        lines.append("| " + " | ".join(str(h) for h in headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for _, row in sub.iterrows():
            lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
        return "\n".join(lines)

    digits_table = _format_table(digits_df)
    wine_table = _format_table(wine_df)

    digits_opt_3d = digits_df[digits_df["Pareto Optimal (3D: Acc, Latency, FLOPs)"]]["Method"].tolist()
    wine_opt_3d = wine_df[wine_df["Pareto Optimal (3D: Acc, Latency, FLOPs)"]]["Method"].tolist()

    report = f"""# Multi-Objective Pareto Frontier Analysis for Green AI

**Research Context:**  
Peykani et al., *“Green Artificial Intelligence: A Comprehensive Review of Metrics, Tools, Challenges, Trends, and Future Prospects”*, Springer, 2026, DOI: [10.1007/s11831-026-10546-2](https://doi.org/10.1007/s11831-026-10546-2).

---

## 1. Executive Summary

Phase 4 establishes a reproducible, deterministic **Multi-Objective Pareto Dominance Framework** evaluating the trade-offs between predictive accuracy, inference latency, theoretical algorithmic complexity (FLOPs), and available energy profiles across six dynamic model selection paradigms:
1. **Decision Tree** (low-capacity, minimal compute baseline)
2. **Neural Network** (high-capacity ceiling)
3. **Dynamic Cascade** (static confidence threshold)
4. **Dynamic Routing** (logistic-regression router)
5. **CAMAS** (Cost-Aware Multi-Criteria Adaptive Selector)
6. **SLA-CAMAS** (budget-constrained adaptive selector across multiple NN invocation limits)

### Evaluated Objective Dimensions:
- **Accuracy (%)** — *Maximize*
- **Inference Latency Median (ms)** — *Minimize*
- **Theoretical FLOPs per Sample** — *Minimize*

---

## 2. Digits Dataset Pareto Analysis (360 Held-Out Test Samples)

### Multi-Objective Frontier Table
{digits_table}

### Key Digits Findings:
1. **Pareto-Optimal Configurations in 3D ({len(digits_opt_3d)}/9):**
   - `{", ".join(digits_opt_3d)}`
2. **SLA-CAMAS Navigates the Efficiency Gap:**
   - Decision Tree operates at **68.33% accuracy, 0.324 ms, 5 FLOPs**.
   - Neural Network operates at **97.22% accuracy, 2.233 ms, 120,128 FLOPs**.
   - SLA-CAMAS @ 20% achieves **72.50% accuracy with 95.0% FLOPs reduction**.
   - SLA-CAMAS @ 35% bridges the gap to **85.56% accuracy with 73.6% FLOPs reduction**.
   - SLA-CAMAS @ 50% achieves **93.61% accuracy** (matching CAMAS) while maintaining **53.0% FLOPs reduction**.
3. **Dominated Configurations in 3D:**
   - `CAMAS (Adaptive Selection)` is technically dominated in 3D by `SLA-CAMAS @ 50%`. Both share identical accuracy (93.61%), identical FLOPs (56,443), and identical test NN usage (46.9%), but SLA-CAMAS @ 50% exhibited slightly lower empirical median latency (2.2050 ms vs 2.2145 ms).

---

## 3. Wine Dataset Pareto Analysis (36 Held-Out Test Samples)

### Multi-Objective Frontier Table
{wine_table}

### Key Wine Findings:
1. **Pareto-Optimal Configurations in 3D ({len(wine_opt_3d)}/9):**
   - `{", ".join(wine_opt_3d)}`
2. **The Cost of Redundant Dynamic Routing Overhead:**
   - `SLA-CAMAS @ 100%` is **strictly dominated by pure Neural Network**. Escalating 100% of samples through SLA-CAMAS adds tree execution and uncertainty scoring overhead (5,969 FLOPs, 0.418 ms) compared to direct Neural Network evaluation (5,952 FLOPs, 0.208 ms) for the exact same 100% accuracy.
3. **Discrete Uncertainty Snapping:**
   - Due to the depth-2 Decision Tree producing only ~3 unique uncertainty values, `SLA-CAMAS @ 0%` and `SLA-CAMAS @ 30%` snap to 0% NN usage. However, because they incur evaluation overhead (17 FLOPs vs 2 FLOPs), they are **strictly dominated** by `Decision Tree` and `Dynamic Cascade`.
4. **CAMAS vs Dynamic Routing Trade-off:**
   - In 2D (Accuracy vs Latency), `CAMAS` dominates `Dynamic Routing` (91.67% accuracy at 0.412 ms vs 0.503 ms).
   - In 2D (Accuracy vs FLOPs), `Dynamic Routing` dominates `CAMAS` (1,826 FLOPs vs 1,835 FLOPs).
   - In 3D (Accuracy, Latency, FLOPs), both remain mutually non-dominated.

---

## 4. Methodological Limitations & Integrity Safeguards

In compliance with the 2026 Springer Green AI Review Paper:

1. **Trade-offs vs Energy Proof:**
   The Pareto frontier characterizes computational and accuracy trade-offs; it does **not** prove physical energy savings on arbitrary hardware.
2. **FLOPs vs Hardware Joules:**
   Theoretical FLOPs represent algorithmic arithmetic complexity. Hardware energy consumption depends heavily on memory bandwidth, CPU cache line hits, and interpreter runtime.
3. **Dataset Scale & Sampling Limits:**
   Both Digits (1,797 samples) and Wine (178 samples) represent small benchmarks. Single train/validation/test splits demonstrate algorithmic feasibility but do not establish universal generalization across large-scale vision, audio, or LLM workloads.
4. **Software Energy Profiling Constraints:**
   CodeCarbon measurements on Windows utilize constant-TDP heuristics rather than hardware RAPL energy counters. Furthermore, sub-3ms inference latency requires multi-trial synthetic batch looping to register above software sampling thresholds (1 Hz).
5. **No Synthetic Composite Scores:**
   We strictly refrain from presenting arbitrary scalar "Green AI Indices" or Energy-to-Accuracy ratios that lack defensible normalization standards.
"""

    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(report)


# --------------------------------------------------------------------------
# Main Orchestrator
# --------------------------------------------------------------------------

def run_pareto_analysis(
    digits_sla_csv: str = "results/sla_camas_digits.csv",
    digits_base_csv: str = "results/enhanced_benchmark.csv",
    wine_sla_csv: str = "results/sla_camas_wine.csv",
    wine_base_csv: str = "results/wine_benchmark.csv",
    output_dir: str = "results",
    figures_dir: str = "results/figures",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes Phase 4 Pareto analysis end-to-end.
    Generates CSVs, summary report, and 4 publication-quality plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Digits Analysis
    digits_out_csv = os.path.join(output_dir, "pareto_digits.csv")
    digits_df = analyze_dataset_pareto(
        dataset_name="Digits",
        sla_csv_path=digits_sla_csv,
        base_csv_path=digits_base_csv,
        output_csv_path=digits_out_csv,
    )

    # 2. Wine Analysis
    wine_out_csv = os.path.join(output_dir, "pareto_wine.csv")
    wine_df = analyze_dataset_pareto(
        dataset_name="Wine",
        sla_csv_path=wine_sla_csv,
        base_csv_path=wine_base_csv,
        output_csv_path=wine_out_csv,
    )

    # 3. Generate Visualizations
    # Digits: Accuracy vs Latency
    plot_pareto_frontier_2d(
        df=digits_df,
        x_col="Latency Median (ms)",
        y_col="Accuracy (%)",
        x_label="Median Inference Latency (ms)",
        y_label="Test Accuracy (%)",
        title="Digits Dataset: Accuracy vs. Inference Latency Pareto Frontier",
        output_path=os.path.join(figures_dir, "digits_accuracy_vs_latency.png"),
        pareto_col="Pareto Optimal (Acc vs Latency)",
        dataset_name="Digits",
    )

    # Digits: Accuracy vs FLOPs (log scale for FLOPs)
    plot_pareto_frontier_2d(
        df=digits_df,
        x_col="Theoretical FLOPs/Sample",
        y_col="Accuracy (%)",
        x_label="Theoretical FLOPs per Sample (log scale)",
        y_label="Test Accuracy (%)",
        title="Digits Dataset: Accuracy vs. Computational Complexity Pareto Frontier",
        output_path=os.path.join(figures_dir, "digits_accuracy_vs_flops.png"),
        pareto_col="Pareto Optimal (Acc vs FLOPs)",
        dataset_name="Digits",
        use_log_x=True,
    )

    # Wine: Accuracy vs Latency
    plot_pareto_frontier_2d(
        df=wine_df,
        x_col="Latency Median (ms)",
        y_col="Accuracy (%)",
        x_label="Median Inference Latency (ms)",
        y_label="Test Accuracy (%)",
        title="Wine Dataset: Accuracy vs. Inference Latency Pareto Frontier",
        output_path=os.path.join(figures_dir, "wine_accuracy_vs_latency.png"),
        pareto_col="Pareto Optimal (Acc vs Latency)",
        dataset_name="Wine",
    )

    # Wine: Accuracy vs FLOPs
    plot_pareto_frontier_2d(
        df=wine_df,
        x_col="Theoretical FLOPs/Sample",
        y_col="Accuracy (%)",
        x_label="Theoretical FLOPs per Sample",
        y_label="Test Accuracy (%)",
        title="Wine Dataset: Accuracy vs. Computational Complexity Pareto Frontier",
        output_path=os.path.join(figures_dir, "wine_accuracy_vs_flops.png"),
        pareto_col="Pareto Optimal (Acc vs FLOPs)",
        dataset_name="Wine",
        use_log_x=False,
    )

    # 4. Summary Markdown Report
    summary_md_path = os.path.join(output_dir, "pareto_summary.md")
    generate_summary_markdown(digits_df, wine_df, summary_md_path)

    return digits_df, wine_df


if __name__ == "__main__":
    print("\n=======================================================")
    print("  PHASE 4: MULTI-OBJECTIVE PARETO FRONTIER ANALYSIS    ")
    print("=======================================================")

    d_df, w_df = run_pareto_analysis()

    print("\n--- DIGITS PARETO ANALYSIS (3D Optimal) ---")
    d_cols = ["Method", "Accuracy (%)", "Latency Median (ms)", "Theoretical FLOPs/Sample", "Pareto Optimal (3D: Acc, Latency, FLOPs)"]
    print(d_df[d_cols].to_string(index=False))

    print("\n--- WINE PARETO ANALYSIS (3D Optimal) ---")
    w_cols = ["Method", "Accuracy (%)", "Latency Median (ms)", "Theoretical FLOPs/Sample", "Pareto Optimal (3D: Acc, Latency, FLOPs)"]
    print(w_df[w_cols].to_string(index=False))

    print("\nGenerated Outputs:")
    print("• results/pareto_digits.csv")
    print("• results/pareto_wine.csv")
    print("• results/pareto_summary.md")
    print("• results/figures/digits_accuracy_vs_latency.png")
    print("• results/figures/digits_accuracy_vs_flops.png")
    print("• results/figures/wine_accuracy_vs_latency.png")
    print("• results/figures/wine_accuracy_vs_flops.png")
