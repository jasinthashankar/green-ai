# Green AI: Dynamic Model Selection

Cascading and routing between a cheap **Decision Tree** and an accurate **Neural Network**, so the expensive model is only used when it is actually needed.

Based on:
- **Cruciani, E. & Verdecchia, R. (2025).** *Choosing to Be Green: Advancing Green AI via Dynamic Model Selection.* Green-Aware AI Workshop @ ECAI 2025.
- **Peykani et al. (2026).** *Green Artificial Intelligence: A Comprehensive Review of Metrics, Tools, Challenges, Trends, and Future Prospects.* Springer. DOI: 10.1007/s11831-026-10546-2.

---

## Idea

Large models cost more time and energy, yet many inputs are easy enough for a small model. Instead of always running the big model, we choose per input:

| Method | How it works |
|---|---|
| **Cascading** | Run the Decision Tree first. If its confidence is at least the threshold (default 0.80), keep its answer. Otherwise ask the Neural Network. |
| **Routing** | A small router (Logistic Regression) reads the tree's output and decides up front whether the Neural Network is needed. |
| **CAMAS** | Cost-Aware Multi-criteria Adaptive Selector. Combines confidence, entropy, and margin into a composite uncertainty score, then routes to the NN only when uncertainty exceeds a validation-calibrated threshold. |
| **SLA-CAMAS** | User specifies a maximum Neural Network invocation budget (e.g. 20%). The threshold is selected on validation data to honour that budget. |
| **Pareto Analysis** | Multi-objective trade-off framework evaluating accuracy maximization against latency and FLOPs minimization, establishing non-dominated operational frontiers. |

**Tasks:**
- **Digits** — classify handwritten digits (scikit-learn `digits`, 1,797 images of 8x8 pixels, 10 classes).
- **Wine** — classify wine cultivars (scikit-learn `wine`, 178 samples, 13 chemical features, 3 classes). Used to validate cross-domain generalization.

---

## Repository Structure

```
green-ai-project/
├── app.py                          # Streamlit dashboard (4 tabs)
├── requirements.txt
├── models/                         # Pretrained Digits models
│   ├── decision_tree.pkl           # DecisionTreeClassifier (depth 5, 64 features)
│   ├── neural_network.pkl          # MLPClassifier (256,128,64,32,16), 60,570 params
│   └── router.pkl                  # LogisticRegression router
├── models_wine/                    # Pretrained Wine models
│   ├── decision_tree.pkl           # DecisionTreeClassifier (depth 2, 13 features)
│   ├── neural_network.pkl          # MLPClassifier (64,32)
│   ├── router.pkl                  # LogisticRegression router
│   └── scaler.pkl                  # StandardScaler fitted on Wine X_train only
├── results/                        # CSVs, markdown summaries, and publication figures
│   ├── final_results.csv           # Original baseline benchmark
│   ├── enhanced_benchmark.csv      # Stage 2 five-method Digits benchmark
│   ├── wine_benchmark.csv          # Wine generalization benchmark
│   ├── combined_benchmark.csv      # Cross-dataset comparison
│   ├── sla_camas_digits.csv        # Phase 3 SLA-CAMAS Digits results
│   ├── sla_camas_wine.csv          # Phase 3 SLA-CAMAS Wine results
│   ├── pareto_digits.csv           # Phase 4 Pareto analysis (Digits)
│   ├── pareto_wine.csv             # Phase 4 Pareto analysis (Wine)
│   ├── pareto_summary.md           # Phase 4 comprehensive summary report
│   └── figures/                    # High-resolution (300 DPI) Pareto figures
│       ├── digits_accuracy_vs_latency.png
│       ├── digits_accuracy_vs_flops.png
│       ├── wine_accuracy_vs_latency.png
│       └── wine_accuracy_vs_flops.png
├── src/
│   ├── dataset.py                  # Digits pipeline: 60/20/20 split, pixels/16
│   ├── dataset_wine.py             # Wine pipeline: 60/20/20 stratified split
│   ├── adaptive_selector.py        # CAMAS implementation
│   ├── sla_camas.py                # SLA-CAMAS: budget-constrained selector
│   ├── enhanced_benchmark.py       # Five-method Digits benchmark
│   ├── wine_experiment.py          # Five-method Wine benchmark
│   ├── sla_benchmark.py            # Six-method SLA-CAMAS benchmark (both datasets)
│   ├── pareto_analysis.py          # Phase 4 multi-objective Pareto frontier engine
│   ├── benchmark.py                # Original two-method comparison
│   ├── model_size_study.py         # NN-size sensitivity study
│   └── threshold_tradeoff.py       # Cascade threshold sensitivity study
└── tests/
    ├── test_adaptive_selector.py   # CAMAS unit tests (6 tests)
    ├── test_enhanced_benchmark.py  # Benchmark unit tests (7 tests)
    ├── test_wine_experiment.py     # Wine experiment unit tests (8 tests)
    ├── test_sla_camas.py           # SLA-CAMAS unit tests (21 tests)
    └── test_pareto_analysis.py     # Pareto analysis unit tests (20 tests)
```

---

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The models were saved with scikit-learn 1.6.1, so `requirements.txt` pins that version.

---

## Run

```powershell
# Original baseline
python src/benchmark.py                  # -> results/final_results.csv
python src/model_size_study.py           # -> results/model_size_study.csv/.png
python src/threshold_tradeoff.py         # -> results/threshold_tradeoff.csv/.png

# Stage 2: five-method benchmark
python src/enhanced_benchmark.py         # -> results/enhanced_benchmark.csv
python src/wine_experiment.py            # -> results/wine_benchmark.csv

# Phase 3: SLA-CAMAS six-method benchmark (both datasets)
python src/sla_benchmark.py              # -> results/sla_camas_digits.csv
                                         #    results/sla_camas_wine.csv

# Phase 4: Multi-objective Pareto frontier analysis (both datasets)
python src/pareto_analysis.py            # -> results/pareto_digits.csv
                                         #    results/pareto_wine.csv
                                         #    results/pareto_summary.md
                                         #    results/figures/*.png

# Tests (62 total, all must pass)
python -m unittest discover tests

# Dashboard
streamlit run app.py
```

---

## Models and Data

### Digits
- **Data split:** 60% train (1,078), 20% validation (359), 20% test (360). Pixels scaled to 0–1 (÷16).
- **Decision Tree:** `DecisionTreeClassifier(max_depth=5)`.
- **Neural Network:** `MLPClassifier` with hidden layers (256, 128, 64, 32, 16), 60,570 parameters, trained in Google Colab. Test accuracy: **97.22%**.
- **Router:** Logistic Regression, trained on validation-set tree probabilities. Label = "tree was wrong but NN was right".

### Wine
- **Data split:** 60% train (106), 20% validation (36), 20% test (36). Stratified by class.
- **Scaler:** `StandardScaler` fitted **only** on X_train, then applied to validation and test sets.
- **Decision Tree:** `DecisionTreeClassifier(max_depth=2)`. Test accuracy: **86.11%**.
- **Neural Network:** `MLPClassifier` with hidden layers (64, 32). Test accuracy: **100%**.
- **Router:** Same approach as Digits.

---

## Results

### Original Baseline (Digits, test set, cascade threshold 0.80)

| Method | Accuracy | NN Usage | Latency Saved vs NN |
|---|---|---|---|
| Decision Tree | 68.3% | 0% | ~86% |
| Neural Network | 97.2% | 100% | baseline |
| **Dynamic Cascade** | 94.2% | 48% | **~23%** |
| **Dynamic Routing** | 92.8% | 47% | **~21%** |

### Stage 2 Enhanced Benchmark (Digits, 5 methods)

| Method | Accuracy | NN Usage | Latency Median | FLOPs Saved |
|---|---|---|---|---|
| Decision Tree | 68.33% | 0% | 0.37 ms | 100% |
| Neural Network | 97.22% | 100% | 2.33 ms | 0% |
| Dynamic Cascade | 94.17% | 48.1% | 2.16 ms | 51.9% |
| Dynamic Routing | 92.78% | 46.7% | 2.34 ms | 53.3% |
| **CAMAS** | **93.61%** | **46.9%** | **2.29 ms** | **53.0%** |

### Wine Generalization Benchmark (5 methods, 36 test samples)

| Method | Accuracy | NN Usage | Latency Median |
|---|---|---|---|
| Decision Tree | 86.11% | 0% | 0.18 ms |
| Neural Network | 100.0% | 100% | 0.24 ms |
| Dynamic Cascade | 86.11% | 0% | 0.14 ms |
| Dynamic Routing | 91.67% | 30.6% | 0.56 ms |
| **CAMAS** | **91.67%** | **30.6%** | **0.48 ms** |

> **Note:** Dynamic Cascade uses a 0.80 confidence threshold. The depth-2 Wine tree always produces confidence ≥ 0.85, so the cascade never fires — it behaves identically to the Decision Tree on this dataset.

### Phase 3: SLA-CAMAS Benchmark (Digits)

SLA-CAMAS lets the user set a maximum Neural Network invocation budget. The threshold is selected on **validation data only** — test labels are never used during calibration.

| Method | SLA Budget | Val NN% | Test Accuracy | Test NN% | FLOPs Saved |
|---|---|---|---|---|---|
| CAMAS (baseline) | — | — | 93.61% | 46.9% | 53.0% |
| SLA-CAMAS | 20% | 5.3% | 72.50% | 5.0% | 95.0% |
| SLA-CAMAS | 35% | 26.7% | 85.56% | 26.4% | 73.6% |
| SLA-CAMAS | 50% | 49.3% | 93.61% | 46.9% | 53.0% |
| SLA-CAMAS | 70% | 67.4% | 96.67% | 66.1% | 33.8% |

### Phase 3: SLA-CAMAS Benchmark (Wine)

| Method | SLA Budget | Val NN% | Test Accuracy | Test NN% | FLOPs Saved |
|---|---|---|---|---|---|
| CAMAS (baseline) | — | — | 91.67% | 30.6% | 69.2% |
| SLA-CAMAS | 0% | 0% | 86.11% | 0% | 99.7% |
| SLA-CAMAS | 30% | 0% | 86.11% | 0% | 99.7% |
| SLA-CAMAS | 50% | 33.3% | 91.67% | 30.6% | 69.2% |
| SLA-CAMAS | 100% | 100% | 100.0% | 100% | −0.3% |

---

## NN-Size Sensitivity (Digits)

| Neural Network | Parameters | Cascade Time Saved | Routing Time Saved |
|---|---|---|---|
| Small (32) | 2,410 | −54.8% | −89.7% |
| Medium (128, 64) | 17,226 | −3.0% | −19.3% |
| Large (256, 128, 64, 32, 16) | 60,570 | 26.9% | 22.1% |
| XL (512, 512, 256) | 429,834 | 47.3% | 49.8% |
| XXL (1024, 1024, 512) | 1,646,090 | 48.1% | 50.0% |

Dynamic selection is only beneficial when the NN is large enough that avoiding it saves meaningful time.

---

## Cascade Threshold Trade-Off (Digits)

| Threshold | Test Accuracy | NN Usage | Time Saved |
|---|---|---|---|
| 0.50 | 91.7% | 43.3% | ~34% |
| 0.80 | 94.2% | 48.1% | ~26% |
| 0.95 | 96.7% | 66.1% | ~13% |

---

## SLA-CAMAS: Findings and Limitations

### How It Works
1. The Decision Tree runs on all input samples and produces class probabilities.
2. A composite uncertainty score is computed from confidence, entropy, and margin.
3. A threshold `tau` is selected on the **validation set** so that the fraction of samples with score ≥ `tau` does not exceed the requested budget `B`.
4. At inference time, samples with score ≥ `tau` are escalated to the Neural Network; the rest use the tree's prediction.

### Key Findings
- **Digits:** Budgets from 20% to 70% are all feasible. A 35% budget achieves 85.6% accuracy (vs 93.6% for full CAMAS) while using only 26.4% of NN calls. A 50% budget matches CAMAS exactly.
- **Wine:** The depth-2 tree produces only ~3 distinct uncertainty values (~0, 0.177, 0.299). Budgets below ~30.6% snap to 0% NN usage (DT-equivalent accuracy of 86.1%). Budgets ≥ 50% snap to 30.6% NN usage (CAMAS-equivalent accuracy of 91.7%).

### Documented Limitations

| Limitation | Detail |
|---|---|
| **Distribution shift** | A validation-selected threshold cannot guarantee the same NN usage rate on the test set. Both validation and test NN fractions are reported for every budget. |
| **Discrete uncertainty values** | Shallow trees (Wine depth-2) produce few distinct probability leaves, limiting the set of achievable NN usage rates. Budgets between discrete levels snap to the nearest feasible rate. |
| **CodeCarbon sensor** | On Windows without RAPL hardware drivers, CodeCarbon uses a constant TDP/CPU heuristic. Energy measurements scale proportionally with runtime and are **estimates**, not physical sensor readings. |
| **FLOPs ≠ energy** | Theoretical FLOPs measure algorithmic operations. Actual hardware energy depends on cache behaviour, memory bandwidth, and SIMD efficiency — factors not captured by FLOPs counts. |
| **Dataset scale** | Wine has only 178 samples (36 test). Results demonstrate algorithmic transferability, not broad generalization across high-dimensional modalities. |
| **No budget guarantee** | SLA-CAMAS selects the best threshold achievable on the validation set. It reports explicitly whether the budget was met on both validation and test; no exact guarantee is claimed. |

---

## Phase 4: Multi-Objective Pareto Frontier Analysis

Phase 4 introduces a deterministic Pareto-dominance algorithm evaluating accuracy maximization alongside latency and computational cost (FLOPs) minimization.

### Objectives Defined
1. **Accuracy (%)** — *Maximize*
2. **Median Inference Latency (ms)** — *Minimize*
3. **Theoretical FLOPs per Sample** — *Minimize*

### Digits Pareto Frontier Summary (9 configurations evaluated)

| Method | Accuracy (%) | Latency (ms) | FLOPs/Sample | Pareto (Acc vs Lat) | Pareto (Acc vs FLOPs) | Pareto (3D Joint) | Dominated By (3D) |
|---|---|---|---|---|---|---|---|
| **Decision Tree** | 68.33% | 0.3240 | 5 | ✓ Yes | ✓ Yes | **✓ Yes** | — |
| **Neural Network** | 97.22% | 2.2328 | 120,128 | ✓ Yes | ✓ Yes | **✓ Yes** | — |
| **Dynamic Cascade** | 94.17% | 2.0660 | 57,733 | ✓ Yes | ✓ Yes | **✓ Yes** | — |
| **Dynamic Routing** | 92.78% | 2.2339 | 56,084 | No | ✓ Yes | **✓ Yes** | — |
| CAMAS | 93.61% | 2.2145 | 56,443 | No | ✓ Yes | No | SLA-CAMAS @ 50% |
| **SLA-CAMAS @ 20%** | 72.50% | 1.1437 | 6,056 | ✓ Yes | ✓ Yes | **✓ Yes** | — |
| **SLA-CAMAS @ 35%** | 85.56% | 1.9764 | 31,750 | ✓ Yes | ✓ Yes | **✓ Yes** | — |
| **SLA-CAMAS @ 50%** | 93.61% | 2.2050 | 56,443 | No | ✓ Yes | **✓ Yes** | — |
| **SLA-CAMAS @ 70%** | 96.67% | 2.4202 | 79,467 | No | ✓ Yes | **✓ Yes** | — |

*Note: CAMAS and SLA-CAMAS @ 50% achieve identical accuracy (93.61%) and FLOPs (56,443) using the same 46.9% NN escalation rate on test data; SLA-CAMAS @ 50% had a marginal 0.0095 ms empirical latency advantage in the repeated trials.*

### Wine Pareto Frontier Summary (9 configurations evaluated)

| Method | Accuracy (%) | Latency (ms) | FLOPs/Sample | Pareto (Acc vs Lat) | Pareto (Acc vs FLOPs) | Pareto (3D Joint) | Dominated By (3D) |
|---|---|---|---|---|---|---|---|
| Decision Tree | 86.11% | 0.1498 | 2 | No | ✓ Yes | No | Dynamic Cascade |
| **Neural Network** | 100.00% | 0.2076 | 5,952 | ✓ Yes | ✓ Yes | **✓ Yes** | — |
| **Dynamic Cascade** | 86.11% | 0.1473 | 2 | ✓ Yes | ✓ Yes | **✓ Yes** | — |
| **Dynamic Routing** | 91.67% | 0.5028 | 1,826 | No | ✓ Yes | **✓ Yes** | — |
| **CAMAS** | 91.67% | 0.4124 | 1,835 | No | No | **✓ Yes** | — |
| SLA-CAMAS @ 0% | 86.11% | 0.2005 | 17 | No | No | No | DT, Cascade, SLA 30% |
| SLA-CAMAS @ 30% | 86.11% | 0.1987 | 17 | No | No | No | DT, Cascade |
| SLA-CAMAS @ 50% | 91.67% | 0.4133 | 1,835 | No | No | No | CAMAS |
| SLA-CAMAS @ 100% | 100.00% | 0.4183 | 5,969 | No | No | No | Neural Network |

*Key Insight on Wine Overhead:* `SLA-CAMAS @ 100%` is **strictly dominated by pure Neural Network**. Escalating 100% of samples through an adaptive wrapper incurs redundant decision-tree execution and uncertainty estimation (5,969 FLOPs, 0.418 ms) vs direct Neural Network execution (5,952 FLOPs, 0.208 ms) for the identical 100% accuracy.

---

## Methodological Boundaries & Limitations

In compliance with the recommendations of the 2026 Springer Green AI Review Paper (*Peykani et al.*):

| Principle / Limitation | Explanation & Boundary |
|---|---|
| **Trade-offs vs. Energy Proof** | The Pareto frontier establishes trade-offs across mathematical objectives; it does **not** constitute proof of physical energy savings. |
| **FLOPs ≠ Joules** | Theoretical FLOPs reduction accounts for arithmetic operations. Physical energy depends on cache hierarchies, SIMD efficiency, memory access, and CPU power states. |
| **Dataset Scale Limits** | Digits (1,797 samples) and Wine (178 samples) provide preliminary algorithmic transferability. Findings cannot be generalized to multi-modal vision or LLMs without domain-specific evaluation. |
| **Latency Measurement Conditions** | Latency distributions are valid only under consistent workloads, repeated warm-up cycles, and controlled background CPU environments. |
| **Energy Profiling Constraints** | CodeCarbon on Windows relies on constant-TDP heuristics without hardware RAPL sensors. Single-batch microsecond inferences require synthetic repeat loops to register above software polling thresholds. |
| **Relativity of Pareto Optimality** | Pareto dominance is strictly defined relative to the specific objective set and evaluated configurations. Adding or removing objectives alters non-dominated memberships. |

---

## Test Suite

```
Ran 62 tests in ~1.6s
OK
```

| Test file | Tests | What is verified |
|---|---|---|
| `test_adaptive_selector.py` | 6 | CAMAS calibration, threshold selection, data hygiene |
| `test_enhanced_benchmark.py` | 7 | FLOPs accounting, latency statistics, asset loading |
| `test_wine_experiment.py` | 8 | Wine split, scaler hygiene, model isolation, FLOPs |
| `test_sla_camas.py` | 21 | Budget edges (0%, 100%), determinism, test-label isolation, infeasibility, model-file integrity, API shape |
| `test_pareto_analysis.py` | 20 | Dominance logic, tie handling, NaN handling, independent frontiers, schema validation, protected file preservation |

---

## Dashboard

`streamlit run app.py` opens six tabs:

1. **Live Comparison:** benchmarks all six methods (Decision Tree, Neural Network, Dynamic Cascade, Dynamic Routing, CAMAS, SLA-CAMAS) on the Digits test set with real-time FLOPs and latency metrics.
2. **Model-Size Study:** how savings change with Neural Network size.
3. **Threshold Trade-Off:** accuracy vs latency as the cascade confidence threshold varies.
4. **Try a Digit:** pick any test sample or upload your own image; see what all 6 strategies predict.
5. **Pareto Frontier Analysis:** interactive Phase 4 Pareto tables and 300 DPI figures for Digits and Wine datasets.
6. **Wine Generalization:** cross-dataset tabular results showing algorithmic transferability.

---

## References

1. Cruciani, E., & Verdecchia, R. (2025). *Choosing to Be Green: Advancing Green AI via Dynamic Model Selection.* Green-Aware AI 2025 @ ECAI 2025.
2. Peykani et al. (2026). *Green Artificial Intelligence: A Comprehensive Review of Metrics, Tools, Challenges, Trends, and Future Prospects.* Springer. DOI: [10.1007/s11831-026-10546-2](https://doi.org/10.1007/s11831-026-10546-2).
