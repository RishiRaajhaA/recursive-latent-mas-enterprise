# B.Tech First Review Presentation Plan (Official Template)

**Review Date:** 19th September 2026  
**Format:** 15 Minutes Presentation + 5 Minutes Viva Q&A (Total 20 Minutes)  
**Template Reference:** [`First_Review_Sample_Presentation.pdf`](file:///d:/FY%20Project/First_Review_Sample_Presentation.pdf)  
**Evaluation Rubrics (100 Marks Total):**
* **C1. Motivation & Problem Statement:** 10 Marks
* **C2. Research Gaps, Objectives & System Architecture:** 15 Marks
* **C3. Methodology Implementation & Working Prototype:** 25 Marks
* **C4. Experimental Results, Analysis & Discussion:** 30 Marks
* **C5. Presentation, Contributions & Viva:** 20 Marks

---

## 📑 Slide-by-Slide Content & Script Plan

---

### Slide 1 | Title Slide
* **Header / Logo:** Amrita Vishwa Vidyapeetham | School of Artificial Intelligence, Coimbatore
* **Slide Title:** FIRST REVIEW – B.TECH PROJECT
* **Project Title:** **Performance and Computational Analysis of Recursive Multi-Agent Systems for Enterprise Process Automation**
* **Team Details:**
  * **Team No.:** D01 / Team 07
  * Rishi Raajha A — `CB.SC.U4AIE23335` (Project Lead)
  * Harshvardhan V — `CB.SC.U4AIE23353`
  * Naveen Sankar R S — `CB.SC.U4AIE23323`
  * Sakthi Pranav S — `CB.SC.U4AIE23336`
* **Supervisor:** Dr. Prem Jagadeesan

---

### Slide 2 | 01 · Project Progress Since Zeroth Review (Criterion C3 / C4 Overview)
* **Core Message:** Show what has been fully built, tested, and validated on benchmark tasks rather than repeating the initial proposal.

```mermaid
flowchart LR
    A["Enterprise Problem\nFormulation"] --> B["Task Stratification\n(114 τ² Tasks into 3 Tiers)"]
    B --> C["Dual Multi-Agent Systems\n(Text ReAct vs Latent ReAct)"]
    C --> D["Full Matrix Execution\n(912 Total Runs\nr = 0, 1, 2, 3)"]
    D --> E["Multi-Metric Analysis\n(+5.0% Hard Acc, -26% Latency\n-25% Tokens)"]

    classDef done fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d;
    class A,B,C,D,E done;
```

* **Measurable Progress Summary:**
  1. **Dataset Stratification:** Classified 114 $\tau^2$-Bench Retail tasks into 3 balanced tiers (38 Easy, 38 Medium, 38 Hard) based on tool complexity and database mutations.
  2. **Dual-Architecture Implementation:** Built and tested both **Multi-Round Text ReAct** (natural language critique loop) and **Recursive Latent ReAct** (continuous latent representation passing with neural adapters).
  3. **Full Matrix Benchmark (100% Completed):** Executed all 912 benchmark evaluations (114 tasks $\times$ 4 depths $r \in \{0, 1, 2, 3\}$ $\times$ 2 architectures) on local GPU hardware.
  4. **Empirical Validation:** Measured accuracy, tool-calling accuracy, latency, token consumption, and computational FLOPs across all runs.

---

### Slide 3 | 02 · Motivation & Problem Statement (Criterion C1 — 10 Marks)

* **Motivation:**
  * Enterprise workflows (such as order management, cancellations, returns, and customer service) require multi-step planning, tool execution, and business rule verification.
  * Standard single-pass LLM agents frequently fail when workflow complexity increases because early errors cannot be corrected.
  * Multi-agent systems can improve reliability through iterative review and reflection, but passing verbose text between agents adds heavy latency and token cost.
  * A recursive latent communication approach allows agents to collaborate directly in vector representation space without generating unnecessary text tokens.

* **Problem Statement:**
  > *"Enterprise automation requires coordinating multiple agent roles to plan, call database tools, and verify policy constraints. Existing text-based multi-agent systems suffer from high token overhead, long response times, and failure accumulation on complex multi-step workflows. This project develops and evaluates a Recursive Multi-Agent System (RecursiveMAS) that uses continuous latent communication, comparing its accuracy, tool-calling precision, latency, and computational efficiency against standard text-mediated multi-agent baselines across varying workflow difficulties."*

* **Scope Boundaries:**
  * **In Scope:**
    * Design and build a 3-agent recursive system (Planner, Tool-Caller, Verifier) on enterprise retail workflows.
    * Compare text-based and latent-based multi-round communication across recursion depths ($r=0, 1, 2, 3$).
    * Evaluate accuracy, tool selection accuracy, wall-clock latency, token count, and FLOPs.
  * **Out of Scope:**
    * Pre-training new foundation models from scratch.
    * Deploying commercial customer-facing web services.

* **Project Objectives:**
  * **Objective 1:** Build a 3-agent collaborative framework (Planner, Tool-Caller, Verifier) connected to an executable enterprise database environment.
  * **Objective 2:** Implement and compare two recursive communication mechanisms: Text ReAct (discrete token feedback) and Recursive Latent ReAct (continuous hidden-state transfer).
  * **Objective 3:** Stratify 114 enterprise tasks into Easy, Medium, and Hard tiers, and evaluate performance scaling across recursion depths $r \in \{0, 1, 2, 3\}$.
  * **Objective 4:** Quantify computational tradeoffs, analyzing accuracy improvements against token savings, latency reductions, and FLOP efficiency.

---

### Slide 4 | 03 · Consolidated Literature Review & Research Gaps (Criterion C2 — 15 Marks)

#### Literature Review Summary Table

| Paper | Methodology | Dataset | Key Findings | Limitations |
|---|---|---|---|---|
| **Recursive Multi-Agent Systems** *(Zou et al., 2026)* | Trains lightweight neural adapters (`RecursiveLink`) enabling agents to pass latent states recursively. | 9 standard benchmarks (MATH500, AIME, GPQA, LiveCodeBench, HotpotQA). | 8.3% higher accuracy, up to 2.4× speedup, and 75.6% token reduction. | Evaluated only on static math and coding tasks; does not evaluate stateful enterprise tools or database workflows. |
| **ReAct: Synergizing Reasoning & Acting** *(Yao et al., 2023)* | Interleaves reasoning thoughts with tool execution and environment observation feedback. | HotpotQA, Fever, ALFWorld, WebShop. | Significantly improves task success and interpretability over standard prompting. | Single-pass execution; early reasoning or tool-call errors cannot be corrected iteratively. |
| **$\tau$-Bench: Tool-Agent-User Benchmark** *(Barres et al., 2024)* | Simulates realistic enterprise environments with dynamic databases, APIs, and business rules. | Retail and Airline domains with stateful customer service workflows. | Establishes realistic pass^k and policy compliance benchmarks for agents. | Only tests single-agent and simple text baselines; does not explore multi-agent latent collaboration. |
| **LoopLM: Scaling Latent Reasoning** *(Zhu et al., 2025)* | Loops Transformer layers to iterate reasoning in latent space without intermediate decoding. | GSM8K, MATH, ARC-Challenge. | Deepens reasoning capacity while cutting down token generation cost. | Restricted to single-model recurrent loops; lacks multi-agent role delegation and tool actions. |

#### Consolidated Research Gaps
* **Gap 1:** Existing recursive multi-agent research has only been evaluated on static question-answering, math, and code generation datasets, leaving its impact on interactive enterprise workflows untested.
* **Gap 2:** Standard multi-agent frameworks on enterprise benchmarks rely entirely on natural language text passing, which incurs heavy latency, high token costs, and verbal circular reasoning loops.
* **Gap 3:** No prior study provides a comprehensive empirical comparison of accuracy, tool-calling precision, latency, and computational FLOPs between text-mediated and latent multi-agent recursion across stratified enterprise task difficulties.

---

### Slide 5 | 04 · Dataset & Task Stratification / Splitting (Criterion C2 / C3 — 15/25 Marks)

* **Enterprise Benchmark Environment ($\tau^2$-Bench Retail):**
  * **Relational Database (SQLite Backend):** Real stateful database containing 500 user profiles, 1,000 product orders, 50 catalog items, and active tracking records.
  * **Executable API Tools (13 Total Tools):** Includes user lookup, order cancellation, address modification, item exchange, item return, refund calculation, and inventory lookup.
  * **Business Policies ($p$):** Explicit business constraints (e.g., 30-day return policy, item condition validation, shipping status checks, address change rules).

* **Task Complexity Metric ($C$):**
  * Task difficulty is calculated using a composite scoring formula:
    $$C = 1.0 \times C_{\text{tool}} + 2.0 \times C_{\text{write}} + 1.0 \times C_{\text{div}}$$
  * Where $C_{\text{tool}}$ = total required tool calls, $C_{\text{write}}$ = count of state-modifying actions (mutations), and $C_{\text{div}}$ = distinct tool diversity.

* **Dataset Split Breakdown (114 Total Tasks):**

| Complexity Tier | Task Count | Criteria & Thresholds | Workflow Characteristics | Example Retail Task |
|---|:---:|---|---|---|
| **Easy** | **38 Tasks** | 1–2 Tool Calls<br/>0 Database Mutations<br/>Low tool diversity | Single-step information retrieval; read-only lookups; zero policy conflicts. | Look up tracking status and estimated delivery date for a customer order by email. |
| **Medium** | **38 Tasks** | 3–4 Tool Calls<br/>1 Database Mutation<br/>Moderate diversity | Sequential tool calls; single database state modification; identity verification required. | Update customer delivery address on a pending order after verifying account details. |
| **Hard** | **38 Tasks** | 5+ Tool Calls<br/>$\ge 2$ Database Mutations<br/>High diversity & dependencies | Multi-table database updates; deep action dependencies; multi-step return and refund calculations. | Process a multi-item return, verify return windows, check replacement inventory, and calculate refund after restocking fee. |

---

### Slide 6 | 05 · Methodology – End-to-End Project Research & Execution Pipeline (Criterion C3 — 25 Marks)

* **Overall Project Methodology Overview:**
  * The project follows a 5-stage systematic engineering and empirical research pipeline, transitioning from enterprise environment setup to task stratification, dual multi-agent engine development, controlled matrix execution, and objective oracle verification.

* **Methodology Pipeline Flowchart:**

```mermaid
flowchart LR
    P1["<b>1. Data & Environment</b><br/>• τ²-Bench Retail Domain<br/>• 13 API Tools & Policies (p)<br/>• Stateful SQLite Backend"] --> P2["<b>2. Task Stratification</b><br/>• Metric C = C_tool + 2C_write + C_div<br/>• 38 Easy Tasks<br/>• 38 Medium Tasks<br/>• 38 Hard Tasks"]
    
    P2 --> P3["<b>3. Dual MAS Engines</b><br/>• <b>Text ReAct:</b> Verbal Critique<br/>• <b>Latent ReAct:</b> Neural Links<br/>  (Rin/Rout, 0 inter-agent tokens)"]
    
    P3 --> P4["<b>4. Matrix Execution</b><br/>• 912 Total Evaluations<br/>• Depths r = 0, 1, 2, 3<br/>• CUDA llama-server (4B LLM)"]
    
    P4 --> P5["<b>5. Oracle Verification</b><br/>• State-based ground truth<br/>• Success Rate, Tool Acc<br/>• Latency, Tokens, FLOPs"]

    classDef p1Box fill:#eff6ff,stroke:#2563eb,stroke-width:1.5px,color:#1e3a8a;
    classDef p2Box fill:#f0fdf4,stroke:#16a34a,stroke-width:1.5px,color:#14532d;
    classDef p3Box fill:#faf5ff,stroke:#9333ea,stroke-width:1.5px,color:#581c87;
    classDef p4Box fill:#fef3c7,stroke:#d97706,stroke-width:1.5px,color:#78350f;
    classDef p5Box fill:#ecfdf5,stroke:#059669,stroke-width:1.5px,color:#064e3b;
    class P1 p1Box;
    class P2 p2Box;
    class P3 p3Box;
    class P4 p4Box;
    class P5 p5Box;
```

* **Detailed 5-Stage Project Pipeline Breakdown:**
  1. **Stage 1 · Dataset & Environment Ingestion:** Ingested the executable $\tau^2$-Bench Retail simulation environment with 13 transactional tools and stateful SQLite database.
  2. **Stage 2 · Complexity Stratification:** Computed composite difficulty scores ($C$) based on tool calls and database mutations, partitioning 114 tasks into 38 Easy, 38 Medium, and 38 Hard.
  3. **Stage 3 · Dual Multi-Agent Engine Implementation:** Built both the **Text ReAct baseline** (text critique feedback) and the **Latent RecursiveMAS architecture** (continuous hidden tensor passing via $R_{\text{in}}$ and $R_{\text{out}}$).
  4. **Stage 4 · Full Matrix Benchmark Execution:** Built an automated execution harness running 912 evaluations locally across all recursion depths ($r=0, 1, 2, 3$) on GPU hardware.
  5. **Stage 5 · Objective Oracle Evaluation:** Verified final database states against enterprise policies and calculated accuracy, tool precision, latency, tokens, and FLOPs.

* **Comparative Summary of the Two Developed Engines:**

| Pipeline Dimension | Text ReAct Baseline Engine | Latent RecursiveMAS Engine (Ours) | Core Advantage |
|---|---|---|---|
| **Inter-Agent Communication** | Natural language text tokens ($150-600\text{ tok/round}$) | Continuous hidden vectors ($h \in \mathbb{R}^{d_h}$) | **$25.2\%$ fewer tokens** |
| **Deliberation Latency** | Full vocabulary projection at each turn | Lightweight neural adapter projections ($R_{\text{out}}$) | **$26.1\%$ faster execution** |
| **Tool Execution Boundary** | Generates text thought $\rightarrow$ text tool call | Keeps reasoning latent $\rightarrow$ decodes JSON at DB boundary | Clear separation of compute |
| **Error Recovery on Hard Tasks** | Vulnerable to text prompt bloat and circular critiques | Continuous representation refinement without token loss | **$+4.9\%$ accuracy on Hard tasks** |

---

### Slide 7 | 06 · Finalized System Architecture & Technical Specifications (Criterion C2 / C3 — 15/25 Marks)

* **1. Multi-Tier System Architecture:**
  * **Tier 1 · Input Ingestion & Context Layer:**
    * Ingests customer queries and dynamic conversational state.
    * Uses **Static KV-Prefix Caching** for invariant context (global system instructions, domain policies $p$, and 13 tool JSON schemas) to avoid recomputing static tokens.
  * **Tier 2 · Multi-Agent Neural Deliberation Engine:**
    * Three specialized agents (Planner $A_1$, Tool-Caller $A_2$, Verifier $A_3$) built on the unified Qwen3.5-4B backbone ($4.2\text{B}$ params, $d_h=2560$, 32 layers).
    * **InnerLink ($R_{\text{in}}$):** 2-layer residual MLP ($R_{\text{in}}(h) = h + W_2 \sigma(W_1 h)$) for intra-agent deliberation state refinement.
    * **OuterLink ($R_{\text{out}}$):** Cross-agent residual adapter ($R_{\text{out}}(h) = W_3 h + W_2 \sigma(W_1 h)$) enabling direct vector communication ($A_1 \xrightarrow{R_{\text{out}}} A_2$ and $A_3 \xrightarrow{R_{\text{out}}} A_1$) without text decoding.
  * **Tier 3 · Enterprise Execution & Environment Layer:**
    * $\tau^2$-Bench Retail domain with stateful SQLite database (500 users, 1,000 orders, 50 catalog items, 13 API tools).
    * Executes lookups, updates, cancellations, exchanges, and refunds with automated validation and state rollback.

* **System Architecture Diagram:**

```mermaid
flowchart TD
    UserReq["<b>Enterprise Customer Request (u)</b>\n(e.g., Return item, change address, lookup order)"] --> CtxMgr["<b>Tier 1: Context & State Manager</b>\n• Static Prefix KV-Cache (Policies p + Tool Schemas)\n• Dynamic Dialogue History"]

    subgraph Tier2 ["Tier 2: Multi-Agent Neural Deliberation Engine (r = 0 .. Depth)"]
        direction TB

        subgraph Agents_Row ["Specialized Agent Nodes (Qwen3.5-4B, dh = 2560)"]
            direction LR
            A1["<b>Planner Agent (A1)</b>\nDecomposes goal into steps\n[InnerLink Rin(h1)]"]
            A2["<b>Tool-Caller Agent (A2)</b>\nConstructs JSON parameters\n[InnerLink Rin(h2)]"]
            A3["<b>Verifier Agent (A3)</b>\nEvaluates policy compliance\n[InnerLink Rin(h3)]"]
        end

        A1 -->|"OuterLink Rout(h1)\n(Continuous Plan Tensor)"| A2
        A3 -.->|"OuterLink Rout(h3)\n(Iterative Feedback: if r < Depth)"| A1
    end

    CtxMgr --> A1

    subgraph Tier3 ["Tier 3: Enterprise Execution & Verification Environment"]
        direction LR
        ToolExec["<b>Tool Execution Sandbox</b>\n• Parameter Validation\n• Safe Transaction Execution"]
        DB[("<b>τ²-Bench Retail SQLite DB</b>\n• 500 Users | 1000 Orders | 50 Items\n• 13 API Tools")]
        ToolExec <--> DB
    end

    A2 -->|"Structured JSON API Call"| ToolExec
    ToolExec -->|"Database Observation State (oi)"| A3
    A3 -->|"Task Completed (r = Depth)\nFinal Text Decoding Head"| Output["<b>Verified Customer Response (y)</b>"]

    classDef ctxNode fill:#f1f5f9,stroke:#64748b,stroke-width:1.5px,color:#0f172a;
    classDef agentNode fill:#eff6ff,stroke:#2563eb,stroke-width:2px,color:#1e3a8a;
    classDef envNode fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#78350f;
    classDef outNode fill:#ecfdf5,stroke:#059669,stroke-width:2px,color:#064e3b;
    class CtxMgr ctxNode;
    class A1,A2,A3 agentNode;
    class ToolExec,DB envNode;
    class Output outNode;
```

* **2. Technical Specifications & Configuration:**

| System Component | Specific Module / Model | Parameters & Dimensions | Functional Role |
|---|---|---|---|
| **Planner Agent ($A_1$)** | `Deliberation-Reflector-Qwen3.5-4B` | $4.2\text{B}$ params, $d_h=2560$, 32 layers, $8\text{k}$ context | Analyzes user goals, breaks down steps, and emits continuous plan tensor $h_1$. |
| **Tool-Caller Agent ($A_2$)** | `Deliberation-Toolcaller-Qwen3.5-4B` | $4.2\text{B}$ params, $d_h=2560$, tuned for JSON API calling | Ingests transformed plan tensor $R_{\text{out}}(h_1)$ and generates executable database API queries. |
| **Verifier Agent ($A_3$)** | `Deliberation-Reflector-Qwen3.5-4B` | $4.2\text{B}$ params, $d_h=2560$, reflection & critique tuned | Checks database observations against business rules; loops feedback tensor $h_3$ or decodes answer. |
| **Inner Neural Link ($R_{\text{in}}$)** | `Deliberation-Innerlinks` | 2-Layer Residual MLP: $R_{\text{in}}(h) = h + W_2 \sigma(W_1 h)$ | Refines representation within each agent between deliberation steps. |
| **Outer Neural Link ($R_{\text{out}}$)** | `Deliberation-Outerlinks` | Cross-agent Residual Adapter: $W_3 h + W_2 \sigma(W_1 h)$ | Projects and aligns latent vectors across heterogeneous agent roles without token generation. |
| **Database Environment** | $\tau^2$-Bench Retail SQLite | 13 API tools, 500 users, 1,000 orders, 50 products | Provides deterministic state tracking, transactional commits, and automated rollback. |
| **Inference Backend** | `llama-server` (CUDA) | FlashAttention, KV-prefix caching, sub-second latency ($\sim 300-800\text{ ms}$) | High-throughput local GPU execution. |

---

### Slide 8 | 07 · Implementation Status – Prototype / Proof-of-Concept (Criterion C3 — 25 Marks)

* **Overall Status:** **100% Phase-1 Prototype Completed & Validated**

| System Module | Functional Area | Current Maturity & Validation Status |
|---|---|:---:|
| **1. Enterprise Task Environment** | 114 $\tau^2$-Bench retail tasks stratified into 3 balanced difficulty tiers with SQLite database integration. | ✅ **100% Completed & Verified** |
| **2. Multi-Agent Framework** | 3 specialized agent roles (Planner, Tool-Caller, Verifier) implemented with structured prompt contexts. | ✅ **Fully Operational** |
| **3. Dual Execution Engines** | Built and validated both Multi-Round Text ReAct (baseline) and Recursive Latent ReAct with neural adapters. | ✅ **Fully Operational** |
| **4. Local GPU Inference Backend** | Configured CUDA backend supporting continuous batching and fast inference on 4B Deliberation models. | ✅ **Active on Hardware** |
| **5. Full Benchmark Matrix Runner** | Automated test suite that executed all 912 experimental runs across all depths ($r=0..3$) and tiers. | ✅ **100% Executed (912/912 Runs)** |
| **6. Data & Visualization Pipeline** | Compiled detailed CSVs, per-run trajectory logs, and comparative evaluation plots. | ✅ **Dataset Compiled & Plotted** |

* **First Review Deliverables Ready:**
  * Fully working multi-agent prototype running locally.
  * Complete empirical benchmark dataset containing 912 full evaluation runs.
  * Publication-ready comparative performance plots.

---

### Slide 9 | 08 · Experimental Setup & Test Matrix (Criterion C3 / C4 — 25/30 Marks)

* **Controlled Experimental Matrix (912 Total Runs):**

```
Total Evaluations: 912 Runs
├── 2 Architectures: Text ReAct (456 Runs) vs. Latent ReAct (456 Runs)
├── 4 Recursion Depths: r = 0 (Single-Pass), r = 1, r = 2, r = 3 (Deep Reflection)
└── 3 Complexity Tiers: Easy (38 Tasks), Medium (38 Tasks), Hard (38 Tasks)
```

* **Evaluated Metrics:**
  1. **Task Success Rate / Accuracy (%):** End-to-end task completion verified by automated database state checks.
  2. **Tool Selection Accuracy (%):** Proportion of API tool calls selecting the correct tool with valid arguments.
  3. **Wall-Clock Latency (Seconds):** Total end-to-end execution time per task.
  4. **Output Token Count:** Number of generated tokens across all deliberation rounds.
  5. **Computational FLOPs (TeraFLOPs):** Total theoretical computation estimated across all agent calls.

---

### Slide 10 | 09 · Results – Quantitative Performance (Criterion C4 — 30 Marks)

#### 1. Recursive Round $r=0$ Evaluation (Single-Pass Baseline)

| Method | Metric | Easy Tier (38) | Medium Tier (38) | Hard Tier (38) | Overall (114) | Improve |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Recursive-TextMAS** *(Text ReAct)* | **Acc.**<br/>**Tool Acc.**<br/>**Time (s)**<br/>**Tokens**<br/>**FLOPs** | $50.3\%$<br/>$10.5\%$<br/>$3.92\text{ s}$<br/>$129$<br/>$7.2\text{ T}$ | $50.2\%$<br/>$73.7\%$<br/>$4.60\text{ s}$<br/>$152$<br/>$8.5\text{ T}$ | $36.5\%$<br/>$92.1\%$<br/>$4.57\text{ s}$<br/>$163$<br/>$8.0\text{ T}$ | $45.7\%$<br/>$58.8\%$<br/>$4.36\text{ s}$<br/>$148$<br/>$7.9\text{ T}$ | Base<br/>Base<br/>Base<br/>Base<br/>Base |
| **RecursiveMAS** *(Latent ReAct)* | **Acc.**<br/>**Tool Acc.**<br/>**Time (s)**<br/>**Tokens**<br/>**FLOPs** | $50.3\%$<br/>$10.5\%$<br/>$2.76\text{ s}$<br/>$92$<br/>$6.4\text{ T}$ | $48.1\%$<br/>$65.8\%$<br/>$3.07\text{ s}$<br/>$103$<br/>$7.5\text{ T}$ | $37.4\%$<br/>$92.1\%$<br/>$3.32\text{ s}$<br/>$113$<br/>$7.3\text{ T}$ | **$45.3\%$**<br/>$56.1\%$<br/>**$3.05\text{ s}$**<br/>**$103$**<br/>**$7.1\text{ T}$** | $-0.4\%$<br/>$-2.6\%$<br/>**$\times 1.43$** *(30% faster)*<br/>**$\downarrow 30.6\%$**<br/>**$\downarrow 10.4\%$** |

#### 2. Recursive Round $r=1$ Evaluation

| Method | Metric | Easy Tier (38) | Medium Tier (38) | Hard Tier (38) | Overall (114) | Improve |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Recursive-TextMAS** *(Text ReAct)* | **Acc.**<br/>**Tool Acc.**<br/>**Time (s)**<br/>**Tokens**<br/>**FLOPs** | $50.3\%$<br/>$10.5\%$<br/>$8.03\text{ s}$<br/>$258$<br/>$17.4\text{ T}$ | $54.7\%$<br/>$72.4\%$<br/>$8.95\text{ s}$<br/>$299$<br/>$21.0\text{ T}$ | $39.6\%$<br/>$90.8\%$<br/>$9.48\text{ s}$<br/>$324$<br/>$18.7\text{ T}$ | $48.2\%$<br/>$57.9\%$<br/>$8.82\text{ s}$<br/>$293$<br/>$19.0\text{ T}$ | Base<br/>Base<br/>Base<br/>Base<br/>Base |
| **RecursiveMAS** *(Latent ReAct)* | **Acc.**<br/>**Tool Acc.**<br/>**Time (s)**<br/>**Tokens**<br/>**FLOPs** | $51.0\%$<br/>$10.5\%$<br/>$6.07\text{ s}$<br/>$196$<br/>$15.3\text{ T}$ | $53.3\%$<br/>$63.2\%$<br/>$6.42\text{ s}$<br/>$216$<br/>$17.5\text{ T}$ | $41.2\%$<br/>$82.9\%$<br/>$7.09\text{ s}$<br/>$248$<br/>$16.7\text{ T}$ | **$48.5\%$**<br/>$52.2\%$<br/>**$6.52\text{ s}$**<br/>**$220$**<br/>**$16.5\text{ T}$** | **$\uparrow 0.3\%$**<br/>$-5.7\%$<br/>**$\times 1.35$** *(26% faster)*<br/>**$\downarrow 25.0\%$**<br/>**$\downarrow 13.4\%$** |

#### 3. Recursive Round $r=2$ Evaluation

| Method | Metric | Easy Tier (38) | Medium Tier (38) | Hard Tier (38) | Overall (114) | Improve |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Recursive-TextMAS** *(Text ReAct)* | **Acc.**<br/>**Tool Acc.**<br/>**Time (s)**<br/>**Tokens**<br/>**FLOPs** | $51.3\%$<br/>$10.5\%$<br/>$12.08\text{ s}$<br/>$383$<br/>$33.1\text{ T}$ | $60.6\%$<br/>$71.1\%$<br/>$13.75\text{ s}$<br/>$443$<br/>$39.0\text{ T}$ | $46.4\%$<br/>$90.4\%$<br/>$14.51\text{ s}$<br/>$474$<br/>$32.7\text{ T}$ | $52.8\%$<br/>$57.3\%$<br/>$13.45\text{ s}$<br/>$433$<br/>$34.9\text{ T}$ | Base<br/>Base<br/>Base<br/>Base<br/>Base |
| **RecursiveMAS** *(Latent ReAct)* | **Acc.**<br/>**Tool Acc.**<br/>**Time (s)**<br/>**Tokens**<br/>**FLOPs** | $53.2\%$<br/>$10.5\%$<br/>$9.14\text{ s}$<br/>$288$<br/>$28.2\text{ T}$ | $57.7\%$<br/>$66.7\%$<br/>$10.03\text{ s}$<br/>$326$<br/>$31.3\text{ T}$ | $49.1\%$<br/>$87.7\%$<br/>$10.77\text{ s}$<br/>$371$<br/>$28.6\text{ T}$ | **$53.3\%$**<br/>$55.0\%$<br/>**$9.98\text{ s}$**<br/>**$328$**<br/>**$29.4\text{ T}$** | **$\uparrow 0.5\%$** *(+2.7% Hard)*<br/>$-2.3\%$<br/>**$\times 1.35$** *(26% faster)*<br/>**$\downarrow 24.3\%$**<br/>**$\downarrow 16.0\%$** |

#### 4. Recursive Round $r=3$ Evaluation (Final Convergence)

| Method | Metric | Easy Tier (38) | Medium Tier (38) | Hard Tier (38) | Overall (114) | Improve |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Recursive-TextMAS** *(Text ReAct)* | **Acc.**<br/>**Tool Acc.**<br/>**Time (s)**<br/>**Tokens**<br/>**FLOPs** | $53.8\%$<br/>$11.2\%$<br/>$16.67\text{ s}$<br/>$512$<br/>$50.8\text{ T}$ | $63.7\%$<br/>$69.1\%$<br/>$19.54\text{ s}$<br/>$597$<br/>$58.7\text{ T}$ | $45.9\%$<br/>$88.2\%$<br/>$19.46\text{ s}$<br/>$654$<br/>$49.0\text{ T}$ | $54.4\%$<br/>$56.1\%$<br/>$18.56\text{ s}$<br/>$588$<br/>$52.8\text{ T}$ | Base<br/>Base<br/>Base<br/>Base<br/>Base |
| **RecursiveMAS** *(Latent ReAct)* | **Acc.**<br/>**Tool Acc.**<br/>**Time (s)**<br/>**Tokens**<br/>**FLOPs** | $57.5\%$<br/>$13.8\%$<br/>$12.89\text{ s}$<br/>$393$<br/>$43.7\text{ T}$ | $60.9\%$<br/>$61.8\%$<br/>$13.93\text{ s}$<br/>$444$<br/>$45.7\text{ T}$ | $50.8\%$<br/>$84.9\%$<br/>$14.68\text{ s}$<br/>$495$<br/>$42.4\text{ T}$ | **$56.4\%$**<br/>$53.5\%$<br/>**$13.83\text{ s}$**<br/>**$444$**<br/>**$43.9\text{ T}$** | **$\uparrow 2.0\%$** *(+4.9% Hard)*<br/>$-2.6\%$<br/>**$\times 1.34$** *(-25.5%)*<br/>**$\downarrow 24.5\%$** *(-84k tok)*<br/>**$\downarrow 16.9\%$** |

---

### Slide 11 | 10 · Results Interpretation & Discussion (Criterion C4 — 30 Marks)

* **Key Results & Takeaways (4 Simple Points):**
  1. **Recursion Boosts Accuracy (+11.1% Gain):**
     * Increasing recursion depth from single-pass ($r=0$) to round $r=3$ improves overall task accuracy from **$45.3\%$ to $56.4\%$** across all 114 benchmark tasks.
     * Iterative verification enables the agent to catch missing arguments and recover from initial execution failures.
  2. **Latent ReAct Outperforms on Hard Workflows (+4.9% on Hard):**
     * On Hard tasks requiring 5+ tool calls and multiple database mutations, Latent ReAct achieves **$50.8\%$ vs. $45.9\%$** for Text ReAct.
     * Continuous latent vectors preserve rich contextual information without getting trapped in repetitive text critique loops.
  3. **Faster Execution Time (26.1% Latency Reduction):**
     * Latent ReAct runs **$26.1\%$ faster** on average ($13.83\text{ s}$ vs. $18.56\text{ s}$ at $r=3$) because it avoids decoding intermediate text tokens between agents.
  4. **Substantial Token & Compute Savings (25.2% Fewer Tokens, 15.6% Lower FLOPs):**
     * Latent ReAct saves **$84,000+$ generated tokens** across the benchmark ($249,610$ vs. $333,558$ tokens) and reduces computational cost from $52.8\text{ T}$ to $43.9\text{ T}$ FLOPs.

---

### Slide 12 | 11 · Prototype / Execution Demonstration (Criterion C3 — 25 Marks)

```mermaid
sequenceDiagram
    autonumber
    actor User as Enterprise Customer
    participant P as Planner (A1)
    participant TC as Tool-Caller (A2)
    participant DB as Retail Database
    participant V as Verifier (A3)

    Note over User,V: Benchmark Case Study: Complex Item Exchange Workflow (Hard Tier)
    User->>P: "I want to exchange delivered item #P104 from Order #W8491"
    
    rect rgb(240, 253, 244)
        Note over P,V: Round 0: Customer Authentication & Order Lookup
        P->>TC: OuterLink(h1) [Continuous Plan Tensor]
        TC->>DB: get_order_details("W8491")
        DB-->>TC: Status: Delivered (Delivery Date: 12 days ago)
        TC->>V: Latent State Handoff
        V-->>P: OuterLink(h3) [Trigger Round 1: Check 30-Day Policy]
    end

    rect rgb(240, 253, 244)
        Note over P,V: Round 1 & 2: Policy Verification & Inventory Check
        P->>TC: OuterLink(h1') -> check_item_inventory("P105")
        TC->>DB: check_inventory(item_id="P105")
        DB-->>TC: Stock: Available (Quantity: 14)
        TC->>V: Latent State Handoff
    end

    rect rgb(236, 253, 245)
        Note over V,User: Round 3: Database Mutation & Response Generation
        TC->>DB: exchange_delivered_order_items(order_id="W8491", old_item="P104", new_item="P105")
        DB-->>V: State Mutation Success: Exchange Order #EX-902 Created
        V->>User: "Your exchange has been processed! Replacement item #P105 is scheduled for shipment."
    end
```

---

### Slide 13 | 12 · Technical Challenges Faced (Criterion C5 — 20 Marks)

* **Key Engineering Challenges Faced:**
  1. **Task Segregation & Difficulty Stratification:**
     * Designing a composite scoring metric ($C = C_{\text{tool}} + 2 C_{\text{write}} + C_{\text{div}}$) to reliably partition 114 enterprise workflows into balanced Easy, Medium, and Hard tiers.
  2. **Code & Runtime Optimization for Local Hardware Configuration:**
     * Serving a 4.2B parameter model locally with persistent CUDA `llama-server`, managing GPU VRAM constraints, continuous batching, and sub-second inference latency.
  3. **Implementation & Routing of Latent ReAct:**
     * Developing `InnerLink` ($R_{\text{in}}$) and `OuterLink` ($R_{\text{out}}$) neural adapters to pass continuous hidden-state tensors ($h \in \mathbb{R}^{d_h}$) between agents without intermediate text decoding.
  4. **Environment State Synchronization & Safe Tool Execution:**
     * Coordinating stateful SQLite database transactions, tool argument validation, and automated rollback mechanisms across multi-round execution passes.

---

### Slide 14 | 13 · Remaining Work & Completion Plan (Criterion C5 — 20 Marks)

* **Phase-2 Remaining Work & Deliverables:**

| Stage | Milestone / Deliverable | Key Tasks & Objectives | Target Timeline |
|:---:|---|---|:---:|
| **1** | **Latent Extraction & Geometric Analysis** | • Extract hidden-state representations ($H_r$) across recursion rounds ($r=0..3$).<br>• Compute representation drift ($D_r$), cosine similarity, and effective rank ($\text{ER}$).<br>• Analyze geometric convergence patterns between successful and failed recovery trajectories. | *Oct – Nov 2026* |
| **2** | **Results Report & Final Presentation** | • Consolidate all empirical benchmark findings, compute efficiency metrics, and geometric plots.<br>• Write and compile the comprehensive final project report / thesis.<br>• Prepare final slide deck and demonstration for the final review viva defense. | *Nov – Dec 2026* |

* **Summary of Next Steps:**
  1. Complete geometric manifold extraction on the compiled 912 evaluation trajectories.
  2. Finalize the project documentation, comprehensive performance report, and viva presentation.

---

### Slide 15 | 14 · Project Contribution & Current Achievements (Criterion C5 — 20 Marks)

#### **Key Achievements Summary (Phase 1)**

| Milestone / Deliverable | Key Achievement & Empirical Impact |
|---|---|
| **1. Enterprise Benchmark Stratification** | Stratified 114 $\tau^2$-Bench Retail tasks into 3 balanced complexity tiers with full database integration. |
| **2. Dual-Architecture Implementation** | Built and tested both Multi-Round Text ReAct (baseline) and Recursive Latent ReAct with neural adapters. |
| **3. Full Matrix Evaluation (912 Runs)** | Completed all 912 benchmark evaluations, achieving $+4.9\%$ accuracy on Hard tasks and $26.1\%$ latency reduction. |
| **4. Multi-Metric Evaluation Framework** | Evaluated task success, tool-calling accuracy, latency, token consumption, and computational FLOPs. |

---

#### **Team Roles & Responsibilities**

| Team Member & Roll No. | Role & Contribution Weight | Core Responsibility & Contribution (1-Point Summary) |
|---|:---:|---|
| **Rishi Raajha A** (`CB.SC.U4AIE23335`) | **Project Lead (Highest Contribution)** | Architected the dual-paradigm multi-agent framework, configured local CUDA inference, and executed the full 912-run benchmark evaluation matrix. |
| **Harshvardhan V** (`CB.SC.U4AIE23353`) | **Core Contributor** | Implemented neural `RecursiveLink` residual adapters and recursion communication dynamics for latent state transfer. |
| **Naveen Sankar R S** (`CB.SC.U4AIE23323`) | **Core Contributor** | Integrated stateful retail database connectors, transaction rollback mechanisms, and telemetry logging pipelines. |
| **Sakthi Pranav S** (`CB.SC.U4AIE23336`) | **Core Contributor** | Conducted literature review, identified research gaps, formulated task stratification scoring metric ($C$), and consolidated documentation. |

---

### Slide 16 | 15 · References (Criterion C5 — 20 Marks)

* **[1]** J. Zou, R. Pan, R. Qiu, P. Lu, S. Diao, J. Jiang, H. Tong, T. Zhang, M. J. Buehler, J. He, and J. Zou, *"Recursive Multi-Agent Systems"*, arXiv preprint, 2026.
* **[2]** S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao, *"ReAct: Synergizing Reasoning and Acting in Language Models"*, International Conference on Learning Representations (ICLR), 2023.
* **[3]** Sierra AI & UC Berkeley, *"$\tau$-Bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains"*, arXiv preprint, 2024.
* **[4]** R.-J. Zhu, Z. Wang, K. Hua, T. Zhang, Z. Li, H. Que, B. Wei, Z. Wen, F. Yin, H. Xing, et al., *"LoopLM: Scaling Latent Reasoning via Looped Language Models"*, arXiv preprint, 2025.
