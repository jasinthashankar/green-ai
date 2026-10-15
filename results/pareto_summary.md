# Multi-Objective Pareto Frontier Analysis for Green AI

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
| Method | SLA Budget (%) | Accuracy (%) | NN Usage (%) [Test] | Latency Median (ms) | Theoretical FLOPs/Sample | Pareto Optimal (Acc vs Latency) | Pareto Optimal (Acc vs FLOPs) | Pareto Optimal (3D: Acc, Latency, FLOPs) | Dominated By (3D) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Decision Tree | — | 68.33 | 0.0 | 0.324 | 5 | ✓ Yes | ✓ Yes | ✓ Yes | — |
| Neural Network | — | 97.22 | 100.0 | 2.2328 | 120128 | ✓ Yes | ✓ Yes | ✓ Yes | — |
| Dynamic Cascade | — | 94.17 | 48.1 | 2.066 | 57733 | ✓ Yes | ✓ Yes | ✓ Yes | — |
| Dynamic Routing | — | 92.78 | 46.7 | 2.2339 | 56084 | No | ✓ Yes | ✓ Yes | — |
| CAMAS (Adaptive Selection) | — | 93.61 | 46.9 | 2.2145 | 56443 | No | ✓ Yes | No | SLA-CAMAS @ 50% |
| SLA-CAMAS @ 20% | 20.0 | 72.5 | 5.0 | 1.1437 | 6056 | ✓ Yes | ✓ Yes | ✓ Yes | — |
| SLA-CAMAS @ 35% | 35.0 | 85.56 | 26.4 | 1.9764 | 31750 | ✓ Yes | ✓ Yes | ✓ Yes | — |
| SLA-CAMAS @ 50% | 50.0 | 93.61 | 46.9 | 2.205 | 56443 | No | ✓ Yes | ✓ Yes | — |
| SLA-CAMAS @ 70% | 70.0 | 96.67 | 66.1 | 2.4202 | 79467 | No | ✓ Yes | ✓ Yes | — |

### Key Digits Findings:
1. **Pareto-Optimal Configurations in 3D (8/9):**
   - `Decision Tree, Neural Network, Dynamic Cascade, Dynamic Routing, SLA-CAMAS @ 20%, SLA-CAMAS @ 35%, SLA-CAMAS @ 50%, SLA-CAMAS @ 70%`
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
| Method | SLA Budget (%) | Accuracy (%) | NN Usage (%) [Test] | Latency Median (ms) | Theoretical FLOPs/Sample | Pareto Optimal (Acc vs Latency) | Pareto Optimal (Acc vs FLOPs) | Pareto Optimal (3D: Acc, Latency, FLOPs) | Dominated By (3D) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Decision Tree | — | 86.11 | 0.0 | 0.1498 | 2 | No | ✓ Yes | No | Dynamic Cascade |
| Neural Network | — | 100.0 | 100.0 | 0.2076 | 5952 | ✓ Yes | ✓ Yes | ✓ Yes | — |
| Dynamic Cascade | — | 86.11 | 0.0 | 0.1473 | 2 | ✓ Yes | ✓ Yes | ✓ Yes | — |
| Dynamic Routing | — | 91.67 | 30.6 | 0.5028 | 1826 | No | ✓ Yes | ✓ Yes | — |
| CAMAS (Adaptive Selection) | — | 91.67 | 30.6 | 0.4124 | 1835 | No | No | ✓ Yes | — |
| SLA-CAMAS @ 0% | 0.0 | 86.11 | 0.0 | 0.2005 | 17 | No | No | No | Decision Tree, Dynamic Cascade, SLA-CAMAS @ 30% |
| SLA-CAMAS @ 30% | 30.0 | 86.11 | 0.0 | 0.1987 | 17 | No | No | No | Decision Tree, Dynamic Cascade |
| SLA-CAMAS @ 50% | 50.0 | 91.67 | 30.6 | 0.4133 | 1835 | No | No | No | CAMAS (Adaptive Selection) |
| SLA-CAMAS @ 100% | 100.0 | 100.0 | 100.0 | 0.4183 | 5969 | No | No | No | Neural Network |

### Key Wine Findings:
1. **Pareto-Optimal Configurations in 3D (4/9):**
   - `Neural Network, Dynamic Cascade, Dynamic Routing, CAMAS (Adaptive Selection)`
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
