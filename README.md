# Geometric and Performance Analysis of Recursive Multi-Agent Systems for Enterprise Process Automation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4+-ee4c2c.svg)](https://pytorch.org/)
[![Benchmark](https://img.shields.io/badge/Benchmark-%CF%84--Bench%20Retail-success.svg)](https://github.com/sierra-research/tau-bench)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation, experimental benchmarks, geometric representation analysis, and thesis artifacts for:

> **Geometric and Performance Analysis of Recursive Multi-Agent Systems for Enterprise Process Automation**  
> *Department of Artificial Intelligence, Amrita Vishwa Vidyapeetham, Coimbatore Campus*

---

## 📌 Overview

Enterprise process automation requires autonomous reasoning agents capable of handling stateful transactions, deterministic API tool execution, and strict corporate business policies. While conventional **Reason-and-Act (ReAct)** and **Reflexion** pipelines rely on verbal natural-language critique loops, iterative multi-agent collaboration suffers from:
1. **Compounding prompt bloat and context degradation** (high token inference cost and latency).
2. **Representational redundancy** (severe inter-agent verbal overlap, $\text{CosSim} > 0.998$).
3. **Mathematical instability** of unconstrained textual feedback loops.

To overcome these bottlenecks, this project evaluates a **Two-Tier Recursive Multi-Agent System** that conducts deliberation directly in **continuous latent representation space** ($h \in \mathbb{R}^{2560}$), strictly decoding tokens at database execution boundaries.

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
├── Report/                              # Complete B.Tech Thesis Documentation
│   ├── mythesis.tex                     # Master LaTeX thesis document
│   ├── mybib.bib                        # Bibliography database
│   ├── mythesis.pdf                     # Compiled final thesis report (37 pages)
│   ├── chapters/                        # Individual thesis chapters (1 to 6 & Appendix)
│   └── images/                          # Publication-quality architectural diagrams & plots
├── requirements.txt                     # Python dependencies
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
@thesis{rishi2026geometric,
  title={Geometric and Performance Analysis of Recursive Multi-Agent Systems for Enterprise Process Automation},
  author={Naveen Sankar, R. S. and Rishi Raajha, A. and Sakthi Pranav, S. and Harshvardhan, V.},
  year={2026},
  school={Department of Artificial Intelligence, Amrita Vishwa Vidyapeetham, Coimbatore Campus},
  note={Under the guidance of Dr. Prem Jagadeesan}
}
```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
