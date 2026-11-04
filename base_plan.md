# Project Brief

## Working Title

**Recursive ReAct for Enterprise Tool-Use: Reliability, Representation Geometry, and Computational Efficiency**

---

# 1. Introduction

## 1.1 Enterprise Agentic Workflows

Enterprise workflows often require more than producing a correct natural-language answer. An agent may need to understand a user's objective, inspect structured information, invoke external tools, maintain state across multiple interactions, perform actions, and verify whether the requested outcome has actually been achieved.

This makes enterprise-style tool-use tasks a useful setting for studying agent reliability. Errors can arise from incorrect planning, selecting the wrong tool, generating incorrect arguments, misunderstanding observations, losing track of state, or stopping before the task is complete.

The proposed project therefore focuses on **agentic reasoning in stateful, tool-mediated enterprise environments**.

## 1.2 ReAct

ReAct (Reason + Act) interleaves reasoning with external actions. Rather than producing a final response immediately, the agent can reason about what information or action is needed, invoke a tool, observe the result, and continue reasoning.

A simplified ReAct trajectory is:

```text
Task
  ↓
Thought
  ↓
Action / Tool Call
  ↓
Observation
  ↓
Thought
  ↓
Action / Tool Call
  ↓
Observation
  ↓
Final Response
```

ReAct provides the first agentic baseline because enterprise tasks naturally involve interactions with external state and tools.

## 1.3 Recursive Reasoning

Ordinary ReAct already contains repeated Thought–Action–Observation steps. These internal steps are therefore not themselves the recursion investigated in this project.

We distinguish:

- **ReAct step:** one Thought → Action → Observation transition.
- **ReAct trajectory:** the complete sequence of ReAct steps used during one attempt.
- **Recursive round:** explicit re-entry into the reasoning process after a complete trajectory, using information from the previous attempt for refinement.

Thus recursive ReAct can be represented as:

```text
ReAct trajectory — Round 1
          ↓
review / refine
          ↓
ReAct trajectory — Round 2
          ↓
review / refine
          ↓
ReAct trajectory — Round 3
          ↓
final result
```

The primary question is whether this additional recursive refinement improves task reliability enough to justify its additional computational cost.

## 1.4 RecursiveMAS

RecursiveMAS extends recursion to multi-agent systems by allowing agents to collaborate through latent representations. Its architecture uses lightweight inner and outer RecursiveLink modules, enabling latent-state refinement within agents and transfer between agents.

The paper describes the multi-agent system as a recursive computation in which the output of the final agent is passed back to the first agent, closing the loop. Intermediate recursive rounds can communicate in latent space and only the final round needs to decode a textual answer.

The paper also compares RecursiveMAS with a text-mediated recursive MAS and evaluates accuracy, inference time, and token usage across recursion depths.

This project uses those ideas as the basis for a latent-recursive enterprise-agent extension.

## 1.5 Problem Motivation

The project follows a controlled progression:

```text
Direct Agent
     ↓
ReAct
     ↓
Recursive ReAct
     ↓
Recursive Latent Multi-Agent System
```

This allows the study to isolate:

1. the effect of explicit reasoning and tool use;
2. the effect of recursive refinement;
3. the effect of recursion depth;
4. the effect of latent versus text-based communication; and
5. the relationship between these architectural choices, reliability, representation geometry, and computational cost.

---

# 2. Enterprise Benchmark and Task Formulation

## 2.1 τ²-Bench Environment

The project uses the **τ-Bench / τ²-Bench** benchmark ecosystem as the primary evaluation environment.

The benchmark provides an executable tool-agent-user setting with domain policies, tools, tasks, environment/database state, and evaluation criteria. This removes the need to construct a new enterprise simulator and allows the main engineering effort to remain focused on the agent architecture and analysis.

The benchmark contains enterprise-style customer-service environments such as retail, airline, and telecom.

## 2.2 Selected Enterprise Domain

The initial preferred domain is **Retail** because it provides a compact stateful environment involving tasks such as order management, returns, and customer/product information.

A representative task can be viewed as:

```text
User request
      ↓
Understand task
      ↓
Retrieve relevant state
      ↓
Perform action / state change
      ↓
Verify result
      ↓
Respond to user
```

The benchmark environment supplies the domain state and tool interfaces, allowing the project to concentrate on reasoning and recursion rather than on implementing enterprise APIs.

The airline and telecom environments remain possible secondary domains for robustness experiments if implementation time permits.

## 2.3 Task Formulation

A benchmark task is represented abstractly as:

\[
T=(u,p,s_0,\mathcal{A},G)
\]

where:

- \(u\) = user request;
- \(p\) = applicable domain policy;
- \(s_0\) = initial environment/database state;
- \(\mathcal{A}\) = available tools/actions;
- \(G\) = task goal/evaluation criteria.

An execution produces a sequence of actions and observations:

\[
T \rightarrow a_1 \rightarrow o_1 \rightarrow a_2 \rightarrow o_2
\rightarrow \cdots \rightarrow y
\]

where actions may include external tool calls and \(o_i\) are observations returned by the environment.

## 2.4 Task Correctness

For tasks involving state changes, correctness will primarily be determined through the benchmark's evaluation criteria and resulting environment state rather than by textual similarity.

Define:

\[
Success(T)=
\begin{cases}
1 & \text{if the task goal/evaluation criteria are satisfied}\\
0 & \text{otherwise.}
\end{cases}
\]

Where richer benchmark reward information is available, it will be reported in addition to binary success.

This makes the task suitable for objective, architecture-level comparison.

## 2.5 Task Complexity

Task complexity is treated as an independent experimental variable.

\[
\boxed{\text{Task Complexity} \neq \text{Recursion Depth}}
\]

A task is considered more complex because it requires more interaction, dependency, tool use, or state manipulation—not because the model is given more recursive computation.

Candidate complexity features are:

### Tool-call count

\[
C_{tool}=N_{\text{tool calls}}
\]

### Interaction depth

\[
C_{turn}=N_{\text{agent/environment turns}}
\]

### Tool diversity

\[
C_{div}=|\{\text{distinct tools used}\}|
\]

### State-change count

\[
C_{state}=N_{\text{state-changing actions}}
\]

### Dependency depth

Length of the longest chain in which one action depends on the result of an earlier action.

A composite complexity score can be considered:

\[
C =
w_1C_{tool}
+w_2C_{turn}
+w_3C_{div}
+w_4C_{state}
+w_5C_{dep}.
\]

The final feature set, weights, and thresholds will be fixed using development/validation data before final test evaluation.

## 2.6 Benchmark Selection

The selected benchmark subset should satisfy:

- multi-step tool interaction;
- meaningful state changes;
- manageable tool vocabulary;
- objective evaluation;
- recoverable failure opportunities;
- enough variation in task interaction complexity.

The preferred benchmark strategy is to keep the original benchmark environment unchanged wherever possible.

The selected tasks will then be divided into:

```text
Low-complexity tasks
Medium-complexity tasks
High-complexity tasks
```

using a complexity definition fixed before test evaluation.

## 2.7 Recovery Subset

A separate subset will be used to study whether recursion enables recovery from an initially unsuccessful attempt.

A recoverable case is one where:

1. the initial execution fails to satisfy the task;
2. the environment remains in a recoverable state; and
3. additional reasoning/tool interaction can still produce a successful outcome.

Define:

\[
Recovery =
\begin{cases}
1 & \text{if a later recursive round succeeds}\\
0 & \text{otherwise.}
\end{cases}
\]

The experiment will record:

- initial outcome;
- final outcome;
- number of recursive rounds;
- additional tool calls;
- additional tokens;
- additional latency.

## 2.8 Task Splitting

The task is the fundamental split unit.

The same underlying task or controlled variants of it should not appear across train and test partitions.

The intended workflow is:

```text
Development tasks
      ↓
select prompts / complexity thresholds / budgets
      ↓
freeze experimental setup
      ↓
held-out test tasks
```

This prevents test information from influencing architectural or benchmark-design decisions.

---

# 3. Agentic Framework

## 3.1 Common Agent Setup

The primary implementation will use a common Qwen backbone across role-specialized agents wherever practical.

Conceptually:

```text
A1 = Qwen + Task Analyst role
A2 = Qwen + Tool / Evidence role
A3 = Qwen + Verification role
```

Using the same backbone reduces confounding from different model architectures and hidden dimensions, which is especially useful for latent representation analysis.

The main model parameters will remain frozen where possible so that the principal experimental differences arise from prompting, recursion, communication medium, and lightweight link modules.

## 3.2 Agent Roles

### A1 — Task Analyst

Responsibilities:

- interpret the user objective;
- identify the relevant state;
- determine what information or action is required;
- form an initial plan;
- identify unresolved issues.

A1 does not require direct tool access in the initial system.

### A2 — Tool / Evidence Agent

Responsibilities:

- determine whether external information/action is needed;
- select the appropriate tool;
- construct tool arguments;
- execute tool calls;
- inspect observations;
- return evidence-backed state information.

A2 is the primary tool-enabled agent.

### A3 — Verification / Decision Agent

Responsibilities:

- assess whether the evidence supports the requested task;
- verify state-changing actions;
- identify unresolved errors or contradictions;
- provide the final decision/result.

Tool access can be restricted to verification tools if the selected domain requires it.

## 3.3 Prompt Structure

Each agent receives:

```text
System instruction
      +
Role-specific instruction
      +
Current user/task information
      +
Tool schema, if applicable
      +
Dynamic state / collaboration information
```

The RecursiveMAS paper explicitly uses a system prompt for all agents together with agent-specific prompts. It also describes downstream agents as receiving their own task context plus transferred latent information.

The implementation therefore distinguishes:

### Static context

- system instruction;
- role instruction;
- fixed tool schema;
- task framing.

### Dynamic context

- tool observations;
- current environment state;
- previous agent state;
- previous recursion-round state.

## 3.4 Tool Calling

Only designated agents will have direct access to tools.

The primary tool-enabled path is:

```text
Agent hidden state
       ↓
tool-call generation
       ↓
external tool
       ↓
observation
       ↓
continued reasoning
```

The tool call is an explicit interface because the environment requires a machine-readable action.

Tool performance will therefore be evaluated independently from final task success.

## 3.5 Latent and Tool Communication

For a latent-recursive agent, one hidden state may support both communication pathways:

```text
                         hidden state h
                           /       \
                          /         \
                         ↓           ↓
                 tool-call output  RecursiveLink
                         ↓           ↓
                       tool       next agent
                         ↓
                    observation
                         ↓
                  updated context
```

This design preserves explicit tool interaction while allowing agent-to-agent reasoning to remain latent.

The project will measure the cost of the explicit tool boundary separately from the communication cost avoided by latent transfer.

## 3.6 Recursive ReAct Structure

Recursive ReAct adds an outer refinement loop around ordinary ReAct:

```text
ReAct trajectory — Round 1
          ↓
review previous reasoning/result
          ↓
ReAct trajectory — Round 2
          ↓
review previous reasoning/result
          ↓
ReAct trajectory — Round 3
          ↓
final result
```

The recursive review should focus on:

- unresolved assumptions;
- incorrect tool use;
- incomplete state understanding;
- contradictory observations;
- whether the previous result satisfies the task.

Additional rounds may invoke tools again when necessary.

## 3.7 Latent Recursive Multi-Agent Structure

The latent system uses:

```text
A1 → latent → A2 → latent → A3
↑                         │
└──────── latent ─────────┘
```

The communication topology therefore has three learned cross-agent transitions:

\[
A_1\rightarrow A_2,\qquad
A_2\rightarrow A_3,\qquad
A_3\rightarrow A_1.
\]

The RecursiveMAS paper uses inner RecursiveLinks for latent refinement within agents and outer RecursiveLinks for cross-agent representation transfer.

## 3.8 RecursiveLink Configuration

The inner link is conceptually a lightweight residual transformation:

\[
R_{in}(h)=h+W_2\sigma(W_1h).
\]

The outer link maps the source representation into the target agent's input space:

\[
R_{out}(h)=W_3h+W_2\sigma(W_1h).
\]

The released RecursiveMAS implementation and checkpoints can be used as a reference implementation. However, the released Deliberation outer-link checkpoint should not automatically be assumed to work unchanged for a new three-agent enterprise topology. Where necessary, topology-specific link adaptation/training will be used.

## 3.9 Static Context and Caching

System prompts, role instructions, and fixed tool schemas are static within an agent configuration.

Where the runtime supports it, these static components can be cached so that recursive rounds do not repeatedly reconstruct identical context.

Caching will be treated as an implementation optimization and reported separately from the conceptual communication mechanism.

---

# 4. Research Questions, Hypotheses and Experimental Design

## 4.1 Research Questions

### RQ1
**Does ReAct improve enterprise task success compared with a direct tool-using agent?**

### RQ2
**Does recursive ReAct improve reliability and recovery from failed or incomplete attempts compared with standard ReAct?**

### RQ3
**How does recursion depth affect task success, tool-use quality, token consumption, latency, and computational cost?**

### RQ4
**Does the benefit of recursive reasoning increase with task interaction complexity?**

### RQ5
**Do hidden-state geometric properties distinguish successful refinement from persistent failure?**

### RQ6
**Can latent recursive multi-agent communication provide recursive-refinement benefits with lower intermediate communication overhead than text-mediated recursion?**

## 4.2 Hypotheses

### H1 — ReAct Effectiveness

ReAct will achieve higher task success than a direct single-pass agent on tasks requiring multi-step tool interaction.

### H2 — Recursive Refinement

Recursive ReAct will improve final task success and recovery over ordinary ReAct on recoverable failures.

### H3 — Complexity Dependence

The benefit of recursive reasoning will be larger on higher-complexity tasks than on low-complexity tasks.

### H4 — Diminishing Returns

Increasing recursion depth will initially improve reliability but eventually produce diminishing gains relative to the additional token, latency, and compute cost.

### H5 — Geometric Signatures

Successful recursive correction will exhibit systematic differences in hidden-state trajectories compared with persistent failure.

### H6 — Latent Efficiency

Latent recursive communication will reduce intermediate text-generation and vocabulary-decoding overhead compared with text-mediated recursive collaboration at matched recursion depth.

## 4.3 Experimental Variables

| Factor | Values |
|---|---|
| Architecture | Direct / ReAct / Recursive ReAct / Recursive Latent MAS |
| Task complexity | Low / Medium / High |
| Recursion depth | 0 / 1 / 2 / 3 |
| Task condition | Normal / Recovery |
| Tool set | Fixed |
| Backbone | Same primary model |
| Environment | Same benchmark environment |

## 4.4 Experimental Comparisons

The main comparisons are:

```text
Direct vs ReAct
```

Tests the contribution of explicit reasoning and acting.

```text
ReAct vs Recursive ReAct
```

Tests the contribution of recursive refinement.

```text
Recursive Text vs Recursive Latent
```

Tests the contribution of the communication medium while keeping recursive structure as consistent as possible.

```text
Low vs Medium vs High Complexity
```

Tests whether recursion becomes more valuable as task demands increase.

```text
r = 1 vs r = 2 vs r = 3
```

Tests how additional recursive computation changes performance and cost.

---

# 5. Geometric Representation Analysis

## 5.1 Hidden-State Representation

Conceptually, an LLM agent performs:

```text
Input tokens / context
        ↓
Transformer
        ↓
Hidden representation h
        ↓
Vocabulary / action projection
        ↓
Token or tool action
```

The project will capture hidden representations before the final vocabulary/action projection where practical.

This is important because the same hidden representation can support both observable outputs and latent communication.

## 5.2 Which Representations Are Extracted

The representation extraction point will be fixed before the main experiment.

Candidate primary representations include:

- final-layer hidden state at a designated decision position;
- pooled representation over a fixed decision context;
- latent thought states exchanged between RecursiveMAS agents.

The primary representation definition should remain consistent across the main experiments.

For a recursive system:

\[
H_{a,r}
\]

denotes the selected representation for agent \(a\) at recursion round \(r\).

## 5.3 Representation Evolution Across Rounds

For a task:

\[
H_0 \rightarrow H_1 \rightarrow H_2 \rightarrow H_3.
\]

The trajectory can be compared across:

- recursion depths;
- architectures;
- complexity levels;
- successful and failed executions.

## 5.4 Geometric Metrics

| Metric | What it measures | Research use |
|---|---|---|
| Cosine similarity | Directional alignment | Alignment between states/agents |
| Representation drift | State change between rounds | Degree of refinement |
| Inter-agent distance | Separation between agents | Specialization vs alignment |
| PCA | Low-dimensional structure | Visualization of trajectories |
| Effective rank | Effective dimensionality | Concentration/diversity |
| Principal angles | Subspace orientation | Similarity of representational subspaces |
| Success–failure separation | Geometric difference between outcome groups | Reliability/failure analysis |

### Cosine similarity

\[
CosSim(H_i,H_j)=
\frac{H_i^\top H_j}
{\|H_i\|\|H_j\|}
\]

Measures the directional similarity between two representations.

### Representation drift

\[
D_r=1-CosSim(H_r,H_{r-1})
\]

Measures how much the representation changes between recursive rounds.

### Inter-agent distance

\[
D_{ij}=\|H_i-H_j\|
\]

Measures separation between agent representations.

### Effective rank

Measures the effective dimensionality of a representation collection.

### Principal angles

Measure the relative orientation of representation subspaces.

## 5.5 Successful vs Failed Trajectories

A central comparison is:

```text
Successful recovery:
H0 → H1 → H2 → correct

Persistent failure:
H0 → H1 → H2 → incorrect
```

The analysis will test whether these trajectories differ in:

- representation drift;
- cosine similarity;
- effective rank;
- PCA path;
- inter-agent distance;
- subspace similarity.

The goal is to determine whether geometric measurements can act as indicators of successful refinement or failure.

## 5.6 Complexity vs Geometry

Compare geometry across:

```text
Low complexity
Medium complexity
High complexity
```

Questions:

- Does task difficulty increase representation drift?
- Does it change effective rank?
- Does it change inter-agent alignment?
- Does the geometric separation between success and failure increase with difficulty?

## 5.7 Recursion Depth vs Geometry

Compare:

\[
r=0,1,2,3
\]

to determine whether increasing recursive depth produces:

- progressive convergence;
- decreasing state changes;
- instability;
- different success/failure signatures.

The RecursiveMAS paper's PCA analysis motivates this type of cross-round study, but this project applies it to enterprise tool-use trajectories and explicitly separates successful recovery from persistent failure.

---

# 6. Performance and Computational Analysis

## 6.1 Task Performance

Primary metrics:

- task success;
- goal satisfaction;
- benchmark reward where available;
- recovery rate;
- failure rate.

## 6.2 Tool-Use Performance

Measure:

- tool-selection accuracy;
- tool-argument correctness;
- valid tool-call rate;
- unnecessary calls;
- repeated calls;
- total tool-call count;
- tool-call failures.

## 6.3 Token Usage

Where the runtime exposes the information, separate:

- reasoning tokens;
- tool-call tokens;
- intermediate agent-communication tokens;
- tool-result/context tokens;
- final response tokens;
- total generated tokens.

The main latent-communication hypothesis concerns the reduction of **intermediate agent-to-agent text generation**, not the elimination of all token usage.

## 6.4 Latency

Measure:

\[
L_{total}
=
L_{model}
+
L_{decode}
+
L_{tool}
+
L_{communication}.
\]

Where possible, record each component separately.

The important comparisons are:

- Direct vs ReAct;
- ReAct vs Recursive ReAct;
- Recursive Text vs Recursive Latent;
- \(r=1\) vs \(r=2\) vs \(r=3\).

## 6.5 FLOPs / Computational Cost

The RecursiveMAS paper gives the main theoretical communication-cost distinction:

### Text-mediated recursion

\[
\Theta(m|V|d_h)
\]

for the vocabulary projection associated with intermediate generated text.

### Latent recursive communication

\[
\Theta(md_h^2)
\]

for RecursiveLink transformation.

The common Transformer forward computation remains in both systems.

The project will use these expressions as the theoretical basis for the communication-cost comparison and distinguish them from empirical hardware measurements.

## 6.6 Empirical Compute

Where practical, measure:

- estimated FLOPs;
- GPU utilization;
- peak GPU memory;
- wall-clock inference time;
- energy or monetary cost proxy.

Theoretical operation complexity and measured hardware/runtime cost will be reported separately.

## 6.7 Accuracy–Cost Tradeoff

The main plots will include:

```text
Task Success vs Tokens
Task Success vs Latency
Task Success vs Estimated Compute
Recovery Rate vs Recursion Depth
```

The objective is to identify when additional recursive computation is worthwhile rather than simply selecting the system with the highest raw accuracy.

---

# 7. Experimental Matrix and Evaluation Protocol

## 7.1 Main Experimental Matrix

| Architecture | r = 0 | r = 1 | r = 2 | r = 3 |
|---|---:|---:|---:|---:|
| Direct | ✓ | — | — | — |
| ReAct | ✓ | — | — | — |
| Recursive ReAct | reference | ✓ | ✓ | ✓ |
| Recursive Latent MAS | reference | ✓ | ✓ | ✓ |

The exact interpretation of \(r=0\) will be kept consistent as the single-pass/non-recursive reference condition.

## 7.2 Complexity Matrix

Every recursive architecture will be evaluated across:

| Task Complexity | r = 1 | r = 2 | r = 3 |
|---|---:|---:|---:|
| Low | ✓ | ✓ | ✓ |
| Medium | ✓ | ✓ | ✓ |
| High | ✓ | ✓ | ✓ |

This allows interaction effects between complexity and recursion depth to be analyzed.

## 7.3 Main Experimental Procedure

For each held-out task:

1. initialize the same benchmark environment state;
2. run the direct baseline;
3. run standard ReAct;
4. run recursive ReAct at \(r=1,2,3\);
5. run latent recursive MAS at matched depths where implemented;
6. record task outcome;
7. record all tool interactions;
8. record tokens and latency;
9. capture selected hidden representations;
10. classify success, failure, or recovery.

## 7.4 Matched-Task Evaluation

The same task and initial environment state should be used across architectures.

Example:

```text
Task 17

Direct          → failure
ReAct           → success
Recursive R1    → failure
Recursive R2    → success
Recursive R3    → success
```

This enables paired per-task analysis.

## 7.5 Recovery Evaluation

For recoverable failures:

```text
Initial attempt
      ↓
failure
      ↓
recursive refinement
      ↓
new tool/action sequence
      ↓
success or persistent failure
```

Record:

- initial outcome;
- final outcome;
- rounds to recovery;
- additional tool calls;
- additional tokens;
- additional latency;
- representation trajectory.

## 7.6 Failure Analysis

Failures should be categorized where possible:

- wrong tool;
- wrong tool arguments;
- missing tool call;
- premature termination;
- policy violation;
- state-tracking error;
- ineffective repeated action;
- failed recovery;
- recursive instability;
- other.

This provides an interpretable explanation for aggregate results.

## 7.7 Repeated Runs

Because LLM inference may be stochastic, important configurations should be evaluated across repeated runs using controlled inference settings.

Report:

- mean;
- standard deviation;
- confidence intervals where appropriate;
- paired comparisons across the same tasks.

## 7.8 Statistical Analysis

The primary analyses should test:

- architecture effects;
- recursion-depth effects;
- task-complexity effects;
- architecture × complexity interaction;
- architecture × recursion-depth interaction;
- geometry differences between success and failure.

Where appropriate, use paired statistical tests because the same benchmark tasks are evaluated across multiple architectures.

---

# 8. Expected Contributions / Outcomes

The project is expected to investigate and potentially contribute:

1. **An empirical evaluation of ReAct in enterprise tool-use tasks.**
2. **A recursive ReAct framework** for iterative refinement and error recovery.
3. **A complexity-sensitive study** of when recursive reasoning is beneficial.
4. **A RecursiveMAS-style latent multi-agent extension** for enterprise tool interaction.
5. **A representation-geometry study** of recursive reasoning, refinement, convergence, and failure.
6. **A computational comparison** involving tool calls, tokens, latency, theoretical communication cost, and estimated FLOPs.
7. **An accuracy–efficiency characterization** showing when additional recursive computation provides useful benefit.

---

# 9. Practical Implementation Scope

The implementation is intentionally constrained so that the major project effort remains in analysis.

### Initial target

- one primary τ²-Bench domain;
- three role-specialized agents;
- one primary tool-enabled agent;
- benchmark-provided tools;
- a common Qwen backbone where practical;
- Direct, ReAct, Recursive ReAct, and latent-recursive variants;
- recursion depths 0–3;
- hidden-state extraction;
- automated trajectory logging;
- task, tool, geometry, and compute evaluation.

### Engineering principle

The benchmark environment should remain as close as possible to the released implementation.

The main engineering effort should therefore be concentrated on:

```text
agent architecture
       +
recursive reasoning
       +
tool-grounded interaction
       +
latent communication
       +
representation analysis
       +
efficiency analysis
```

rather than on developing a new enterprise simulator.

---

# 10. Key Terminology

**Tool-use task**  
A task requiring the agent to interact with one or more external tools or APIs.

**ReAct step**  
One Thought → Action → Observation transition.

**ReAct trajectory**  
The complete sequence of ReAct steps used for one task execution.

**Recursive round**  
One complete execution of the selected outer reasoning structure followed by re-entry using information from the previous round.

**Recursive ReAct**  
A ReAct trajectory that is explicitly reviewed and repeated across multiple outer rounds.

**Latent communication**  
Transfer of hidden representations between agents instead of decoding intermediate collaboration into text.

**Tool boundary**  
The explicit interface between an agent and an external environment/tool.

**Representation drift**  
Geometric change between hidden representations across states or recursion rounds.

**Recovery**  
Successful task completion after an initially unsuccessful execution through additional recursive refinement.

**Task complexity**  
Difficulty determined from the interaction and state requirements of the benchmark task, independent of recursion depth.

---

# 11. Current Working Research Statement

> **This project investigates whether recursive reasoning-and-acting improves the reliability and efficiency of LLM agents performing multi-step enterprise tool-use tasks. Using τ-Bench / τ²-Bench as an executable enterprise-style environment, the study compares direct agents, ReAct, recursive ReAct, and RecursiveMAS-style latent multi-agent systems under controlled task complexity and recursion depth. Performance is evaluated through task success, tool-use correctness, error recovery, token consumption, latency, and computational cost, while hidden-state geometry is analyzed to determine whether representation dynamics provide measurable indicators of successful refinement and failure.**
