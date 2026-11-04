# Geometric and Performance Analysis of Recursive Multi-Agent Systems for Enterprise Process Automation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4+-ee4c2c.svg)](https://pytorch.org/)
[![Benchmark](https://img.shields.io/badge/Benchmark-%CF%84--Bench%20Retail-success.svg)](https://github.com/sierra-research/tau-bench)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation, experimental benchmarks, geometric representation analysis, and publication artifacts for:

> **Geometric and Performance Analysis of Recursive Multi-Agent Systems for Enterprise Process Automation**  
> *Author: Rishi Raajha A*

---

## 💡 Key Novelties & Scientific Contributions

While recent advances in autonomous LLM agents explore verbal reflection or static latent looping, this project introduces several key novelties at the intersection of continuous neural communication, multi-agent systems, and representation geometry:

1. **First Continuous Latent Communication in Stateful, Tool-Governed Environments:**
   * Prior latent reasoning research (e.g., RecursiveMAS, LoopLM) was strictly evaluated on static math or code generation benchmarks without external side-effects.
   * **Novelty:** This work is the first to implement and benchmark continuous latent-state multi-agent communication within a **stateful, transactional enterprise domain** ($\tau^2$-Bench Retail), managing relational databases (SQLite), deterministic API tools, and multi-condition corporate business rules.

2. **Two-Tier Topology with Boundary-Only Token Decoding:**
   * Traditional multi-agent systems (e.g., AutoGen, Multi-Round ReAct) force agents to talk via natural language critique strings, causing severe context bloat and compounding hallucinations.
   * **Novelty:** We decouple intermediate deliberation from surface token generation. The Planner ($A_1$), Tool-Caller ($A_2$), and Verifier ($A_3$) communicate through continuous hidden activation tensors ($h \in \mathbb{R}^{2560}$) via lightweight `InnerLink` and `OuterLink` neural adapters. Discrete token generation is strictly restricted to external database execution boundaries, slashing latency by **26.1%**, token generation by **25.2%**, and computational FLOPs by **15.6%**.

3. **Geometric Manifold Formulation of Agent Reasoning Trajectories:**
   * Agent evaluation has historically been limited to macroscopic behavioral outputs (pass/fail).
   * **Novelty:** We formulate recursive multi-agent deliberation mathematically as trajectory flows across a high-dimensional manifold ($\mathcal{M} \subset \mathbb{R}^{2560}$). We mathematically track step-wise drift ($D_r$), Participation Ratio ($\text{PR}$), and Centered Kernel Alignment (CKA) across 912 benchmark runs ($r \in \{0, 1, 2, 3\}$).

4. **Empirical Trajectory Stabilization & Subspace Compression without Collapse:**
   * **Novelty:** We mathematically and statistically prove that Latent ReAct exhibits **>70% drift contraction** into a stable localized attractor basin ($p = 1.9 \times 10^{-20}$ via paired Wilcoxon signed-rank test; robust at $p = 7.1 \times 10^{-15}$ under duplicate-filtered controls), whereas Text ReAct wanders indefinitely ($\approx 0.034$/step, $p = 0.96$). Furthermore, both architectures demonstrate **20–23% subspace compression** ($18.9 \to 15.2$ dimensions) while preserving necessary capacity without dimensional collapse.

5. **Role Orthogonality vs. Redundancy:**
   * **Novelty:** In text-mediated multi-agent pipelines, appending shared dialogue histories causes severe representational redundancy ($\text{CosSim} > 0.998$). We demonstrate that learned `OuterLink` adapters preserve **near-orthogonal functional specialization** across Planner, Tool-Caller, and Verifier personas ($\text{CosSim} \le 0.087, D_{ij} \approx 51.5$).

6. **Rigorous Failure Diagnostics & Honest Null Result Discovery:**
   * **Novelty:** Through non-parametric hypothesis testing with Holm-Bonferroni correction ($m=10$), we uncover an honest null result: global role-averaged system states ($H_r$) do not linearly predict task failure ($p = 0.54$). Instead, execution failures originate from localized, tool-specific parameterization errors ($H_2$), offering crucial diagnostic clarity for industrial AI deployment.

---

## 📌 Architecture Overview

```
                    ┌────────────────────────────────────────────────────────┐
                    │  Tier 2: Multi-Agent Neural Deliberation Engine (r)    │
                    │                                                        │
[User Request (u)] ─┼─► Planner (A₁) ──OuterLink (R_out)──► Tool-Caller (A₂) │
                    │        ▲                                      │        │
                    │        │ OuterLink (Feedback)                 ▼        │
                    │    Verifier (A₃) ◄──── Database Obs  [JSON API Calls]  │
                    └────────┬──────────────────────────────────────┼────────┘
                             │ (r = Terminal Depth)                 │
                             ▼                                      ▼
                   [Verified Response (y)]               [SQLite Execution Sandbox]
```

---

## 📚 Base Papers & References

This research builds upon and extends foundational work in latent reasoning, multi-agent architectures, and stateful enterprise benchmarking:

1. **RecursiveMAS (Primary Base Paper):**
   * *Recursive Multi-Agent Systems with Latent Deliberation*  
     Authors: Xiyuan Yang, Jiaru Zou, Rui Pan, Ruizhong Qiu, Pan Lu, Shizhe Diao, et al. (2026).  
     🔗 **Paper:** [arXiv:2604.25917](https://arxiv.org/abs/2604.25917)  
     🔗 **Codebase:** [RecursiveMAS GitHub Repository](https://github.com/RecursiveMAS/RecursiveMAS)

2. **$\tau$-Bench / $\tau^2$-Bench (Enterprise Benchmark Base Paper):**
   * *$\tau$-Bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains*  
     Authors: Shunyu Yao, Noah Shinn, Pedram Razavi, Karthik Narasimhan (Sierra & Princeton, 2024).  
     🔗 **Paper:** [arXiv:2406.12045](https://arxiv.org/abs/2406.12045)  
     🔗 **Official Repository:** [sierra-research/tau-bench](https://github.com/sierra-research/tau-bench)

3. **ReAct Paradigm:**
   * *ReAct: Synergizing Reasoning and Acting in Language Models*  
     Authors: Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao (ICLR 2023).  
     🔗 **Paper:** [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)

4. **Reflexion:**
   * *Reflexion: Language Agents with Verbal Reinforcement Learning*  
     Authors: Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, Shunyu Yao (NeurIPS 2023).  
     🔗 **Paper:** [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)

5. **LoopLM (Latent Looping Foundations):**
   * *Scaling Latent Reasoning via Looped Language Models*  
     Authors: Rui-Jie Zhu, Zixuan Wang, Kai Hua, Tianlong Zhang, Zhiyuan Li, et al. (2025).  
     🔗 **Paper:** [arXiv:2510.25741](https://arxiv.org/abs/2510.25741)

6. **Neural Representation Geometry:**
   * *Neural Representational Geometry of Concepts in Large Language Models*  
     Authors: Linden Schrage, Kazuki Irie, Haim Sompolinsky (NeurIPS NeurReps 2024).

---

## 📦 Datasets & Models Download Links

### 1. Enterprise Benchmark Dataset ($\tau^2$-Bench Retail)
The evaluation environment uses the stateful **Retail Domain** of Sierra's $\tau$-Bench benchmark:
* **Database:** SQLite relational schema containing **500 user profiles**, **1,000 multi-item orders**, product inventories, and return/refund tracking records.
* **Tools:** 13 deterministic Python APIs (order lookups, status updates, returns, exchanges, refunds, discount validations).
* **Task Benchmark:** 114 enterprise customer service tasks stratified into balanced **Easy** ($C \le 2$), **Medium** ($3 \le C \le 5$), and **Hard** ($C \ge 6$) complexity tiers across 912 total execution trajectories ($r \in \{0, 1, 2, 3\}$).
* **Download / Setup:** Clone the benchmark directly from the official Sierra repository:
  ```bash
  git clone https://github.com/sierra-research/tau-bench.git tau2-bench-main
  pip install -e ./tau2-bench-main
  ```

### 2. Foundation Model Backbone
All experiments utilize a unified foundation model backbone to isolate architectural and routing improvements from base capacity:
* **Model Family:** **Qwen3.5-4B** / **Qwen2.5-Coder** ($P = 4.4 \times 10^9$ parameters, $d_h = 2560$, 32 hybrid layers).
* **Hugging Face Model Repositories:**
  * [Qwen Models on Hugging Face](https://huggingface.co/Qwen)
  * [Qwen2.5-3B-Instruct on Hugging Face](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
  * [Qwen2.5-Coder Series](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)
* **Quantization & Execution:** Quantized to 4-bit medium block precision (`Q4_K_M` GGUF, 2.74 GB VRAM footprint) using `llama-cpp-python` with FlashAttention-2 and cuBLAS. Neural adapters ($R_{\text{in}}, R_{\text{out}}$) operate in full unquantized floating-point precision (`bfloat16`/`float32`).

---

## 🚀 Key Experimental Findings

| Metric | Multi-Round Text ReAct | Recursive Latent ReAct (Ours) | Relative Improvement |
| :--- | :---: | :---: | :---: |
| **Terminal Goal Accuracy ($r=3$)** | 54.4% | **56.4%** | **+2.0 pp** |
| **Hard Complexity Tier Accuracy** | 45.9% | **50.8%** | **+4.9 pp** |
| **Clean Successes (Zero Errors)** | 30 tasks | **33 tasks** | **+10.0%** |
| **Error Recovery Yield** | 24.6% (28 tasks) | **28.9% (33 tasks)** | **+17.5%** |
| **Mean End-to-End Latency** | 11.30 s | **8.35 s** | **-26.1%** |
| **Total Generated Tokens** | 67,034 tokens | **50,622 tokens** | **-25.2%** |
| **Computational Inference FLOPs** | 13,076 TFLOPs | **11,038 TFLOPs** | **-15.6%** |
| **Enterprise Policy Violations** | 13 tasks | **10 tasks** | **-23.1%** |

### Geometric Insights:
* **Rapid Trajectory Stabilization:** Latent ReAct concentrates 64% of displacement into the initial round ($0.065$) followed by a **>70% drift contraction** down to $0.018$ ($p = 1.9 \times 10^{-20}$ via Wilcoxon signed-rank test; robust at $p = 7.1 \times 10^{-15}$ under duplicate filtering). Text ReAct wanders continuously ($\approx 0.034$/step, $p = 0.96$).
* **Subspace Compression without Collapse:** Effective dimensionality compresses by **20–23%** ($18.9 \to 15.2$ dimensions) using the Participation Ratio, retaining capacity for multi-step reasoning.
* **Role Specialization:** OuterLink adapters enforce near-orthogonal functional differentiation ($\text{CosSim} \le 0.087, D_{ij} \approx 51.5$) versus severe textual overlap ($\text{CosSim} > 0.998$).
* **Honest Null Result:** Macroscopic pooled states ($H_r$) do not linearly predict task success ($p = 0.54$ after Holm-Bonferroni correction); execution failures correlate with localized tool-parameter schemas ($H_2$).

---

## 🛠️ Repository Structure

```
├── experiments/                         # Complete benchmark runners & execution matrix
│   ├── engine_recursive_react.py        # Core execution engine (Text vs. Latent ReAct)
│   ├── run_full_evaluation_matrix.py    # 912 execution trajectory runner
│   ├── classify_benchmark_tasks.py      # Composite complexity metric (Easy/Med/Hard)
│   ├── data_splits/                     # Stratified task configurations
│   └── evaluation_metrics/              # Output tables, FLOP counters, latency logs
├── project/implementation1/             # Geometric representation analysis engine
│   ├── extract_latent_representations.py# Layer-32 hidden state extraction pipeline
│   ├── compute_geometry_metrics.py      # Drift, Participation Ratio, CKA & CosSim
│   ├── analyze_error_geometry_correlation.py # Failure mode & linear probing diagnostics
│   ├── generate_geometry_plots.py       # Trajectory attractors & PCA manifold visualizations
│   └── results/                         # Summary CSVs, JSON logs, and figures
├── requirements.txt                     # Python dependencies
├── RecursiveMAS.pdf                     # Foundational base paper reference
└── README.md                            # Project documentation
```

---

## ⚡ Quickstart Guide

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/RishiRaajhaA/recursive-latent-mas-enterprise.git
cd recursive-latent-mas-enterprise

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Clone and install tau-bench
git clone https://github.com/sierra-research/tau-bench.git tau2-bench-main
pip install -e ./tau2-bench-main
```

### 2. Running Benchmarks
```bash
# Run full 912 evaluation matrix across depths r = {0, 1, 2, 3}
python experiments/run_full_evaluation_matrix.py

# Classify tasks into complexity tiers
python experiments/classify_benchmark_tasks.py
```

### 3. Extracting Geometric Manifolds & Computing Metrics
```bash
# Extract high-dimensional latent representations (dh = 2560)
python project/implementation1/extract_latent_representations.py

# Compute drift, Participation Ratio, and subspace angles
python project/implementation1/compute_geometry_metrics.py

# Analyze error modes and correlation probes
python project/implementation1/analyze_error_geometry_correlation.py

# Generate publication plots
python project/implementation1/generate_geometry_plots.py
```

---

## 📜 Citation

If you find this work or codebase helpful in your research, please cite:

```bibtex
@article{rishi2026geometric,
  title={Geometric and Performance Analysis of Recursive Multi-Agent Systems for Enterprise Process Automation},
  author={Rishi Raajha, A.},
  year={2026}
}
```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
