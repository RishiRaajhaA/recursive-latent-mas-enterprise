# Geometric Analysis Output Directory (`outputs/`)

This directory contains all computed artifacts, presentation-ready figures, statistical tables, and interpretation reports for the geometric analysis of **RecursiveMAS (Latent ReAct vs. Text ReAct)** across 114 $\tau^2$-Bench Retail tasks.

---

## Directory Structure

```text
outputs/
├── README.md               # This document (inventory & directory guide)
├── FINDINGS.md             # Comprehensive scientific inferences, statistical tests, and caveats
├── results.json            # Full numeric telemetry from initial exploratory run
├── final/                  # Clean, uncluttered, presentation-ready single-focus figures
│   ├── fig_A_convergence.png    # Step-wise state drift (Latent settles, Text wanders)
│   ├── fig_B_dimensionality.png # Intrinsic dimensionality (Participation Ratio across rounds)
│   ├── fig_C_outcome_link.png   # Effect size forest plot (Success vs. Failure geometry)
│   └── results_final.json       # Exact statistics, bootstrap CIs, and p-values for final figures
├── figures/                # Exploratory multi-panel diagnostics
│   ├── fig0_performance_anchor.png  # Task success and accuracy across depths & tiers
│   ├── fig1_effective_rank.png       # Effective rank & participation ratio by agent & tier
│   ├── fig2_drift.png                # Multi-panel drift, tortuosity, and path length
│   ├── fig3_cross_agent.png          # Cross-agent raw/centered cosine & linear CKA heatmaps
│   ├── fig4_pca_trajectories.png     # 2D PCA trajectories across rounds and difficulty tiers
│   └── fig5_outcome_probes.png       # Linear probe decodability for success and task tier
└── tables/                 # Tabular CSV data for reports and LaTeX integration
    ├── drift_steps.csv               # Mean step-wise cosine distances across agents
    ├── effective_rank.csv            # Participation ratio & entropy rank by agent and round
    └── success_vs_failure_geometry.csv # Statistical comparisons (Cliff's delta, Mann-Whitney U, Holm p)
```

---

## 1. Presentation Figures (`outputs/final/`)

These figures were redesigned to be clean, publication-ready, and uncluttered (no overlapping text boxes, clean axes, and high contrast):

### `fig_A_convergence.png`: State Drift per Recursion Step
* **Metric**: Cosine distance ($1 - \cos(\mathbf{h}^{(r)}, \mathbf{h}^{(r-1)})$) between consecutive-round system states.
* **Key Finding**: Latent ReAct concentrates **64%** of its total movement in the very first transition ($r_0 \to r_1$, drift $= 0.065$) and immediately contracts to a near-static manifold ($0.018 \to 0.019$, Wilcoxon contraction $p = 1.9 \times 10^{-20}$). Text ReAct, by contrast, drifts steadily ($0.038 \to 0.032 \to 0.033$).
* **Robustness Check**: Even when removing all rollouts with duplicate/cached states, the latent contraction holds ($p = 7.1 \times 10^{-15}$), whereas text drift becomes completely flat ($p = 0.96$).

### `fig_B_dimensionality.png`: Dimensionality of System State
* **Metric**: Participation Ratio ($PR = (\sum \lambda_i)^2 / \sum \lambda_i^2$) computed over 200 random subsets of $n=90$ tasks.
* **Key Finding**: Latent system representations occupy fewer effective dimensions than text representations across all rounds ($18.9 \to 15.2$ for Latent vs. $22.0 \to 16.8$ for Text). Both architectures compress their representational subspace by $\approx 20\text{--}23\%$ as recursion proceeds.

### `fig_C_outcome_link.png`: Success vs. Failure Effect Sizes
* **Metric**: Cliff's Delta ($\delta$) effect size with 95% bootstrap confidence intervals across 5 candidate geometric predictors (net drift, path length, final-step drift, vector norm, tortuosity).
* **Key Finding**: Honest negative result. All confidence intervals span zero or border it, and no feature achieves statistical significance after Holm-Bonferroni correction (minimum $p_{\text{adj}} = 0.54$). Global geometry does not predict downstream task success.

---

## 2. Statistical Tables (`outputs/tables/`)

* **`drift_steps.csv`**: Contains step-by-step drift metrics for each agent role ($H_1, H_2, H_3, H_r$) across both architectures.
* **`effective_rank.csv`**: Contains full-sample and subsampled ($n=90$) participation ratios and entropy-based effective rank values for all 4 recursion rounds.
* **`success_vs_failure_geometry.csv`**: Full comparison table between successful and failed rollouts, reporting mean values, Cliff's delta, Mann-Whitney $U$ test statistics, raw $p$-values, and Holm-adjusted $p$-values.

---

## 3. Telemetry JSONs

* **`results_final.json`**: Exact numerical output for the 3 final figures (step means, bootstrap CIs, contraction $p$-values, participation ratios, and outcome effect sizes).
* **`results.json`**: Comprehensive diagnostic dump from the exploratory pipeline, including cross-agent CKA matrices, PCA variance ratios, and cross-validated linear probe AUC scores.

---

## 4. How to Regenerate

To recompute all metrics and regenerate figures:
```bash
# Generate the 3 clean presentation figures and final stats:
python final_figures.py

# Generate exploratory multi-panel diagnostics and CSV tables:
python geometric_analysis.py
```
