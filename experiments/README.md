# Experimental Implementation Framework

This folder houses the modular experimental pipeline for **Geometric and Performance Analysis of Recursive Multi-Agent Systems for Enterprise Process Automation**, matching [plan.md](file:///d:/FY%20Project/plan.md).

```
experiments/
├── agents/         # Agent role definitions (A1 Analyst, A2 Tool Agent, A3 Verifier) & prompts
├── models/         # Qwen model loader (4-bit/8-bit/bf16) & RecursiveLink modules (R_in, R_out)
├── runners/        # Execution engines (ReAct baseline, Text-Recursive, Latent RecursiveMAS)
├── geometry/       # Representation extraction & geometric metrics (Drift, Effective Rank, PCA, Subspace)
├── results/        # Evaluation logs, benchmark scores, and metric exports
└── utils/          # Token counters, latency profilers, and SQLite tau2-bench wrappers
```
