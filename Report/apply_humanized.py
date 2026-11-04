import sys, re

# Read Humanized.txt
with open('Report/Humanized.txt', 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')

h_dict = {
    'ch3_p4_1': lines[2].replace('Paragraph 1 ', ''),
    'ch3_p4_2': lines[4],
    'ch3_p5': lines[7],
    'abs_1': lines[11],
    'abs_2': lines[13],
    'intro_p4_1': lines[17],
    'intro_p4_2': lines[19],
    'intro_p5_1': lines[21],
    'intro_p5_2': lines[23],
    'intro_p7': lines[25],
    'intro_p8': lines[27],
    'intro_p9': lines[29],
    'intro_p10_34': lines[31],
    'intro_p11': lines[33],
    'ch2_p4': lines[37],
    'ch2_p5': lines[39],
    'ch2_p67': lines[41],
    'ch2_p8': lines[43],
    'ch2_p9_1': lines[45],
    'ch2_p9_2': lines[47],
    'ch2_p1213': lines[49],
    'ch2_p14': lines[51],
    'ch2_p15_1': lines[53],
    'ch2_p15_2': lines[55],
    'ch3_p6': lines[59].replace('Figure 3.1 caption: ', ''),
    'ch3_p7': lines[63],
    'ch3_p16': lines[65],
    'ch3_p19': lines[69]
}

# 1. Update mythesis.tex
with open('Report/mythesis.tex', 'r', encoding='utf-8') as f:
    mythesis = f.read()

orig_abs_1 = (
    "Enterprise process automation requires autonomous reasoning agents capable of handling stateful transactions, "
    "deterministic application programming interface invocations, and organizational policy constraints. "
    "While Reason-and-Act frameworks separate reasoning traces from environment actions, verbal critique loops in "
    "multi-step processes cause cumulative prompt bloat, significant token generation costs, and compounding execution errors. "
    "This thesis presents the design, geometric manifold characterization, and empirical benchmark evaluation of a "
    "Recursive Multi-Agent System that operates on continuous latent representations. "
    "The system rigidly decodes at database execution borders and routes continuous hidden tensors across recursive rounds. "
    "Deliberation is divided among specialized Planner, Tool-Caller, and Verifier agents connected via neural residual adapters. "
    "It is powered by a frozen Qwen3.5-4B foundation model."
)
assert orig_abs_1 in mythesis, 'orig_abs_1 not found in mythesis'
mythesis = mythesis.replace(orig_abs_1, h_dict['abs_1'], 1)

orig_abs_2 = (
    "Inter-agent analysis shows near-orthogonal functional specialization in Latent ReAct "
    "($\\text{CosSim} \\le 0.087, D_{ij} \\approx 51.5$), in contrast to significant verbal overlap in Text ReAct "
    "($\\text{CosSim} > 0.998$). Statistical analysis establishes an honest null result, indicating that rather than "
    "global averaged states that linearly differentiate task performance, localized, tool-specific representations "
    "may merit additional research as predictors of execution reliability."
)
assert orig_abs_2 in mythesis, 'orig_abs_2 not found in mythesis'
mythesis = mythesis.replace(orig_abs_2, h_dict['abs_2'], 1)

with open('Report/mythesis.tex', 'w', encoding='utf-8') as f:
    f.write(mythesis)
print('mythesis.tex updated successfully.')

# 2. Update chapters/introduction.tex
with open('Report/chapters/introduction.tex', 'r', encoding='utf-8') as f:
    intro = f.read()

orig_intro_p4 = (
    "Enterprise operational workflows like customer service, logistics fulfillment, warranty processing, "
    "and inventory management are increasingly dependent on autonomous decision-making systems. "
    "In a stateful industrial setting, an AI agent needs to do more than just produce free-form natural language responses. "
    "Instead, the agent must call external Application Programming Interfaces, verify organizational business policies, "
    "deconstruct complex customer objectives, verify relational database states, handle transaction updates, "
    "and ensure that all terminal restrictions are met."
)
assert orig_intro_p4 in intro, 'orig_intro_p4 not found'
intro = intro.replace(orig_intro_p4, h_dict['intro_p4_1'] + '\n\n' + h_dict['intro_p4_2'], 1)

orig_intro_p5_6 = (
    "The Reason-and-Act paradigm established a key basis for agentic intelligence by combining sequential "
    "reasoning trails with actions and observations in the actual environment. Instead of generating responses "
    "in a single feed-forward run, an agent generates concepts, selects tools, and absorbs observations iteratively. "
    "However, single-pass execution might go wrong when handling complex multi-step processes. Initial tool selection "
    "errors, parameter formatting problems, or hasty assumptions might lead to compounding policy violations or task failure.\n\n"
    "Modern multi-agent systems offer specialized role delegation and recursive reflection to reduce early execution "
    "mistakes. However, text-mediated communication is the only method used in conventional systems. Agents communicate "
    "their intentions, execution logs, critiques, and error messages using normal natural language strings. In long-term "
    "business tasks, this verbal approach causes fundamental operational challenges. Context windows collect a "
    "substantial quantity of token history with each recursive critique round, which can increase operational inference "
    "costs and result in latency degradation. Furthermore, the absence of mathematical contractivity guaranties in "
    "unrestricted natural language feedback may cause agents to wander around the prompt space or oscillate between "
    "conflicting explanations without arriving at a definitive response."
)
assert orig_intro_p5_6 in intro, 'orig_intro_p5_6 not found'
intro = intro.replace(orig_intro_p5_6, h_dict['intro_p5_1'] + '\n\n' + h_dict['intro_p5_2'], 1)

orig_intro_p7 = (
    "To overcome these structural limitations, this work investigates recursive multi-agent architectures that "
    "interact directly within continuous latent representation spaces. Through parameterized neural adapters that "
    "route continuous hidden tensors between specialized agent roles, deliberation takes place within the "
    "high-dimensional internal geometry of the model. Natural language decoding is only allowed in areas where "
    "external database Application Programming Interfaces are utilized."
)
assert orig_intro_p7 in intro, 'orig_intro_p7 not found'
intro = intro.replace(orig_intro_p7, h_dict['intro_p7'], 1)

orig_intro_p8 = (
    "Multi-agent language systems that depend on text-mediated reflection in intricate enterprise workflows have "
    "substantial token overhead, high inference delay, and compounding execution errors as task complexity rises. "
    "The literature lacks an empirical and geometric description of continuous latent-state communication in stateful, "
    "tool-mediated environments.\n"
    "This thesis describes the design, implementation, and testing of a Recursive Latent Multi-Agent System against "
    "the conventional Multi-Round Text ReAct in an enterprise customer service domain. It investigates whether latent "
    "recursive communication improves task completion reliability, reduces computing cost, and demonstrates empirical "
    "trajectory stabilization within limited solution zones without dimensional collapse."
)
assert orig_intro_p8 in intro, 'orig_intro_p8 not found'
intro = intro.replace(orig_intro_p8, h_dict['intro_p8'], 1)

orig_intro_p9 = (
    "Stratifying 114 enterprise retail tasks into balanced Easy, Medium, and Hard tiers on $\\tau^2$-Bench "
    "across 912 trajectories; calculating geometric manifold metrics to mathematically characterize trajectory "
    "stabilization; evaluating runtime failure diagnostics using predictive linear probing; and creating a "
    "modular two-tier multi-agent architecture with specialized Planner, Tool-Caller, and Verifier agents "
    "powered by a unified frozen Qwen3.5-4B foundation model coupled via neural residual adapters."
)
assert orig_intro_p9 in intro, 'orig_intro_p9 not found'
intro = intro.replace(orig_intro_p9, h_dict['intro_p9'], 1)

orig_intro_p10_34 = (
    "Third, we demonstrate that, in contrast to the severe context overlap seen in verbal critique passing "
    "($\\text{CosSim} > 0.998$), continuous latent routing via neural adapters maintains near-orthogonal functional "
    "role specialization across agent personas ($\\text{CosSim} \\le 0.087, D_{ij} \\approx 51.5$).\n"
    "Lastly, after multiple-hypothesis correction, our failure analysis yields an honest null result: global, "
    "role-averaged representations do not linearly predict downstream task success. This suggests that localized, "
    "tool-specific representations, rather than macroscopic geometric displacement, may merit more research as "
    "execution reliability predictors."
)
assert orig_intro_p10_34 in intro, 'orig_intro_p10_34 not found'
intro = intro.replace(orig_intro_p10_34, h_dict['intro_p10_34'], 1)

orig_intro_p11 = (
    "The remainder of this thesis is organized in this manner. Chapter 2 reviews the foundational literature "
    "on representation geometry, recursive self-reflection, multi-agent frameworks, and ReAct. Chapter 3 presents "
    "the neural adapters, experimental design, and system architecture. Chapter 4 presents the geometric and error "
    "analysis of latent representations. Chapter 5 presents benchmark performance evaluations, complexity scaling, "
    "and geometric visualizations. Chapter 6 concludes the thesis by summarizing the key findings, constraints, "
    "and possible directions for further investigation. Appendix A provides a description of the benchmark API "
    "tools and relational database structure."
)
assert orig_intro_p11 in intro, 'orig_intro_p11 not found'
intro = intro.replace(orig_intro_p11, h_dict['intro_p11'], 1)

with open('Report/chapters/introduction.tex', 'w', encoding='utf-8') as f:
    f.write(intro)
print('chapters/introduction.tex updated successfully.')

# 3. Update chapters/chapter2.tex
with open('Report/chapters/chapter2.tex', 'r', encoding='utf-8') as f:
    ch2 = f.read()

orig_ch2_p4 = (
    "Yao et al. \\cite{yao2023react} developed the Reason-and-Act (ReAct) paradigm in foundational research "
    "to combine reasoning traces with tangible environmental activities. By combining natural language reasoning "
    "with executable commands executed in external software environments, ReAct demonstrated notable advantages "
    "over standard chain-of-thought prompting on multi-hop question answering and decision-making benchmarks. "
    "ReAct's primary benefit is that, by grounding the model's internal talks on actual environmental data, "
    "it lessens open-ended hallucinations."
)
assert orig_ch2_p4 in ch2, 'orig_ch2_p4 not found'
ch2 = ch2.replace(orig_ch2_p4, h_dict['ch2_p4'], 1)

orig_ch2_p5 = (
    "However, standard ReAct and verbal reflection frameworks are still fundamentally limited by the nature of "
    "natural language communication. When an agent encounters an inaccurate database return or selects an improper "
    "Application Programming Interface schema early in operation, later verbal critiques frequently become verbose, "
    "inconsistent, or fixated on previous errors. The context window is filled by linguistic critique strings that "
    "collect autoregressively, resulting in compounding failure loops that impede task completion."
)
assert orig_ch2_p5 in ch2, 'orig_ch2_p5 not found'
ch2 = ch2.replace(orig_ch2_p5, h_dict['ch2_p5'], 1)

orig_ch2_p67 = (
    "The cognitive load and instability of single-agent pipelines on complicated reasoning activities have led "
    "to the widespread use of multi-agent systems. Wu et al. \\cite{wu2023autogen} developed AutoGen, a modular "
    "architecture that enables multi-agent communication across specialized personas. Planners, programmers, "
    "and code reviewers are examples of specialized functional agents in these systems that break down complex "
    "operational goals and interact via structured prompt passing.\n\n"
    "Although multi-agent specialization improves issue decomposition, conventional solutions simply use "
    "text-mediated communication. Agents must serialize intermediate plans, status summaries, and mistake critiques "
    "into clear natural language tokens so that subsequent agents may parse and re-encode them. In stateful business "
    "tasks that require regular iterative collaboration, this verbal message paradigm creates severe operational "
    "constraints. Repeated dialog history expands the prompt context with each recursive cycle, leading to "
    "substantial inference latency degradation and high computational cost. Furthermore, agents frequently oscillate "
    "between conflicting explanations without arriving at a definitive response due to the mathematical instability "
    "of unconstrained natural language feedback."
)
assert orig_ch2_p67 in ch2, 'orig_ch2_p67 not found'
ch2 = ch2.replace(orig_ch2_p67, h_dict['ch2_p67'], 1)

orig_ch2_p8 = (
    "To assess autonomous agents in realistic enterprise environments, interactive simulators with programmable "
    "APIs, ordered states, and realistic organizational constraints are required. Yao et al. \\cite{sierra2024tau} "
    "developed the $\\tau$-Bench benchmark, which established a standardized benchmark environment for evaluating "
    "tool-agent-user interactions in real-world enterprise domains such as retail customer service and airline "
    "reservations. Unlike static question-answering datasets, $\\tau$-Bench provides an executable environment based "
    "on relational databases, explicit corporate business policies, and user persona simulators."
)
assert orig_ch2_p8 in ch2, 'orig_ch2_p8 not found'
ch2 = ch2.replace(orig_ch2_p8, h_dict['ch2_p8'], 1)

orig_ch2_p9 = (
    "The benchmark demonstrates that even the most sophisticated proprietary models suffer considerable reliability "
    "erosion while negotiating multi-step dependency workflows that include database alterations, strict policy "
    "verification, and long conversational turns. For instance, when transactions require to confirm return "
    "eligibility periods, check inventory levels, or deduct restocking charges, models frequently display policy "
    "breaches or produce incorrect tool arguments. Because of this, $\\tau$-Bench provides an ideal, exacting testbed "
    "for studying recursive agents' dependability."
)
assert orig_ch2_p9 in ch2, 'orig_ch2_p9 not found'
ch2 = ch2.replace(orig_ch2_p9, h_dict['ch2_p9_1'] + '\n\n' + h_dict['ch2_p9_2'], 1)

orig_ch2_p1213 = (
    "To understand these internal representations, researchers have used techniques from high-dimensional brain geometry. "
    "Schrage, Irie, and Sompolinsky \\cite{schrage2024neural} described the neural representational geometry of "
    "concepts in large language models, demonstrating how neural networks arrange semantic categories and related "
    "concepts into structured, low-dimensional geometric manifolds. By proving that cognitive changes in deep networks "
    "can be mathematically represented as trajectory flows inside representation manifolds, their research showed that "
    "geometric metrics provide direct interpretability into internal network dynamics.\n\n"
    "Furthermore, Sorscher et al. \\cite{sorscher2022beyond} and Roy and Vetterli \\cite{roy2007effective} formalized "
    "geometric metrics such as the Participation Ratio and effective rank to quantify the continuous dimensionality "
    "spanned by neural manifolds, providing mathematical tools to assess subspace compression and detect catastrophic "
    "dimensional collapse. Concurrently, Bj{\\\"o}rck and Golub \\cite{bjorck1973numerical} may compute primary angles "
    "between linear subspaces using conventional geometric formulas, offering a rigorous method of monitoring manifold "
    "alignment and separability across various agent roles and task execution phases."
)
assert orig_ch2_p1213 in ch2, 'orig_ch2_p1213 not found'
ch2 = ch2.replace(orig_ch2_p1213, h_dict['ch2_p1213'], 1)

orig_ch2_p14 = (
    "A comprehensive synthesis of the surveyed literature reveals a glaring lack of coherence, even though "
    "AutoGen \\cite{wu2023autogen} formalizes agent specialization, $\\tau$-Bench \\cite{sierra2024tau} provides the "
    "stateful enterprise benchmark, RecursiveMAS \\cite{zou2026recursive} and LoopLM \\cite{zhu2025looplm} demonstrate "
    "the efficacy of latent recursive deliberation, and neural geometry \\cite{schrage2024neural, roy2007effective} "
    "provides mathematical tools for manifold analysis."
)
assert orig_ch2_p14 in ch2, 'orig_ch2_p14 not found'
ch2 = ch2.replace(orig_ch2_p14, h_dict['ch2_p14'], 1)

orig_ch2_p15 = (
    "As far as we can tell from the reviewed literature, continuous latent recursive multi-agent deliberation in "
    "stateful enterprise tool-use systems governed by relational databases and strict business policy constraints "
    "has not been investigated. Previous latent reasoning studies largely focused on static algorithmic benchmarks "
    "or single-model looped pre-training, which did not address the challenges of deterministic API tool selection, "
    "multi-item state updates, and policy verification. Furthermore, the examined literature does not provide an "
    "empirical description of the internal manifold geometry of agent hidden states across recursive passes to "
    "evaluate enterprise agent execution issues. This thesis closes this specific research gap by implementing a "
    "two-tier latent multi-agent architecture on the stateful $\\tau^2$-Bench Retail domain, benchmarking it against "
    "Multi-Round Text ReAct over 912 execution rollouts, and providing a comprehensive geometric characterization "
    "of the underlying representation manifolds."
)
assert orig_ch2_p15 in ch2, 'orig_ch2_p15 not found'
ch2 = ch2.replace(orig_ch2_p15, h_dict['ch2_p15_1'] + '\n\n' + h_dict['ch2_p15_2'], 1)

with open('Report/chapters/chapter2.tex', 'w', encoding='utf-8') as f:
    f.write(ch2)
print('chapters/chapter2.tex updated successfully.')

# 4. Update chapters/chapter3.tex
with open('Report/chapters/chapter3.tex', 'r', encoding='utf-8') as f:
    ch3 = f.read()

orig_ch3_p4 = (
    "The suggested business system's two-tier multi-agent architecture, which deconstructs complex decision-making "
    "processes, consists of three specialized functional agents. The first tier consists of the Planner Agent, denoted "
    "by $A_1$. Deconstructing high-level user intent, identifying operational constraints, cross-referencing corporate "
    "business policies, and developing thorough execution plans are some of its responsibilities. The second tier contains "
    "the execution and verification components. After consuming the continuous plan representations generated by the "
    "Planner, the Tool-Caller Agent (designated as $A_2$) maps planned actions to specific Application Programming "
    "Interface schemas to produce structured JavaScript Object Notation arguments. The Verifier Agent, designated as "
    "$A_3$, analyzes policy compliance, monitors the outcome provided by the external environment, assesses if the "
    "database state changes meet the customer's request, and generates corrective feedback signals in the event that "
    "mistakes or inconsistencies occur."
)
assert orig_ch3_p4 in ch3, 'orig_ch3_p4 not found'
ch3 = ch3.replace(orig_ch3_p4, h_dict['ch3_p4_1'] + '\n\n' + h_dict['ch3_p4_2'], 1)

orig_ch3_p5 = (
    "Figure~\\ref{fig:sys_arch} illustrates the complete system architecture and compares the operational flow of "
    "conventional Multi-Round Text ReAct with the continuous tensor routing of Recursive Latent ReAct. In the "
    "text-mediated baseline, each communication step between agents requires the autoregressive creation of natural "
    "language critique tokens, leading to high inference latency and substantial context bloat. Conversely, "
    "Recursive Latent ReAct restricts intermediate deliberation to continuous representation space by strictly decoding "
    "tokens at the database execution boundary."
)
assert orig_ch3_p5 in ch3, 'orig_ch3_p5 not found'
ch3 = ch3.replace(orig_ch3_p5, h_dict['ch3_p5'], 1)

orig_ch3_p6 = (
    "\\caption{System Architecture: Two-tier agent architecture that compares Multi-Round Text ReAct (verbal critique loops) "
    "with Recursive Latent ReAct (continuous tensor routing via neural adapters).}"
)
assert orig_ch3_p6 in ch3, 'orig_ch3_p6 caption not found'
ch3 = ch3.replace(orig_ch3_p6, "\\caption{" + h_dict['ch3_p6'] + "}", 1)

orig_ch3_p7 = (
    "All comparative results isolate architectural design rather than base capacity thanks to the Qwen3.5-4B model "
    "\\cite{qwen2026team} ($P = 4.4 \\times 10^9$ total parameters)."
)
assert orig_ch3_p7 in ch3, 'orig_ch3_p7 not found'
ch3 = ch3.replace(orig_ch3_p7, h_dict['ch3_p7'], 1)

orig_ch3_p16 = (
    "The database design models 500 distinct customer user profiles, 1000 multi-item client orders, extensive "
    "product catalogs with inventory tiers, warehouse distribution locations, and return tracking records. "
    "Among the executable tools available are user identification, order lookups, pending order updates, item "
    "returns, product exchanges, refund calculations, and validations of promotional discounts. The atmosphere "
    "is controlled by strict corporate business procedures, such as those that prohibit returns after thirty days, "
    "prohibit modifications to orders that have already been delivered, and charge restocking fees on specific item categories."
)
assert orig_ch3_p16 in ch3, 'orig_ch3_p16 not found'
ch3 = ch3.replace(orig_ch3_p16, h_dict['ch3_p16'], 1)

orig_ch3_p19 = (
    "The per-depth and per-tier accuracy metrics displayed in Table~\\ref{tab:depth_progression} and "
    "Table~\\ref{tab:complexity_tier} represent the mean continuous goal satisfaction accuracy averaged across all "
    "tasks in each category, which explains why these values show continuous fractional percentages rather than "
    "discrete step increments.\n"
    "Binary task success, denoted by the boolean flag $\\text{is\\_success}$, is granted when the target goal "
    "satisfaction score hits or above a 70\\% barrier ($S_{\\text{goal}} \\ge 0.70$)."
)
assert orig_ch3_p19 in ch3, 'orig_ch3_p19 not found'
ch3 = ch3.replace(orig_ch3_p19, h_dict['ch3_p19'], 1)

with open('Report/chapters/chapter3.tex', 'w', encoding='utf-8') as f:
    f.write(ch3)
print('chapters/chapter3.tex updated successfully.')
print('ALL 4 FILES UPDATED CLEANLY!')
