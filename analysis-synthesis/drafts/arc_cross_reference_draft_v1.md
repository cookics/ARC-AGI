# Shared Difficulty, Different Burdens

## A Cross-Reference Draft Across Humans, LLMs, Non-LLM Systems, and Solver Structure in ARC

## Abstract

This draft consolidates the main empirical and interpretive threads already developed in this repository: LLM psychometrics, human ARC testing, non-LLM comparisons, approved-solver complexity analysis, cross-ARC latent linkage, efficiency analysis, and the recent small-model ARC literature. The central picture is coherent but not trivial. Humans and LLMs do appear to share a real ARC difficulty axis, yet they do not appear to load on that axis in the same way. The strongest result in the current project is that structural program complexity tracks LLM task difficulty far more strongly than human difficulty. Human difficulty, by contrast, looks more tied to time cost, search burden, and substantial pair-level heterogeneity within the same task. The non-LLM systems are the most disappointing part of the current evidence if the goal is a clean human-like alternative: they add complementary signal and rescue a few human-easy, LLM-hard items, but they do not yet reproduce the human difficulty profile as well as the LLM consensus does. At the same time, recent non-LLM ARC papers point to a mechanistic direction that still matters: strong inductive bias, iterative refinement, and test-time candidate selection. The right synthesis is therefore not that one family has solved ARC "the human way." It is that ARC appears to contain a shared abstraction factor plus source-specific burdens, with LLMs looking especially sensitive to final solver structure, humans looking more search- and interaction-sensitive, and current non-LLM systems occupying an interesting but still underpowered middle ground.

## 1. Introduction

The repository now contains multiple ARC papers and addenda that were produced for different immediate questions: whether ARC behaves like a psychometric instrument for LLMs, whether human item difficulty can be estimated reliably from sparse logs, whether solver complexity predicts difficulty, whether non-LLM systems look more human-like, whether ARC-AGI-1 and ARC-AGI-2 can be linked, and whether recent small-model ARC papers share a deeper mechanistic through-line. Each workstream answered something real, but the project now needs a single draft that says what the whole body of evidence adds up to.

The most important reason to consolidate these threads is that they are not independent. The human paper changes how the LLM psychometric story should be interpreted. The solver-complexity work changes how we think about "difficulty" itself. The non-LLM paper complicates any simple humans-versus-LLMs framing. The latent cross-ARC work clarifies where the human evidence is stable and where it is not. The efficiency analysis reveals which tradeoffs track performance across source families. And the external paper synthesis suggests that recent non-LLM progress is converging on a common computational loop rather than on one magic architecture.

The strongest version of the emerging thesis is as follows:

ARC contains a shared difficulty structure across humans and machine systems, but that shared structure is incomplete. LLM difficulty is especially sensitive to the structural complexity of the final successful solver. Human difficulty is less structure-loaded and more tied to search effort, time cost, and pair-level variation inside the same task. Current non-LLM systems contribute real but limited complementary signal and should not yet be described as more human-like than the LLM consensus on the data currently in hand.

That is the story this draft tries to make explicit.

## 2. Source Corpus and Questions

This draft synthesizes seven connected source streams:

1. The LLM psychometric manuscript in `analysis-llm-psychometrics`, including the ARC paper and the broader benchmark-manifold export package.
2. The human testing paper in `analysis-human/papers/human-testing`, plus the supplementary split-half and alignment notes in `analysis-human/creme-analysis`.
3. The non-LLM psychometric paper in `analysis-non-llm/papers/psychometric`.
4. The solver-complexity master paper and companion reports in `analysis-python-complexity`.
5. The cross-benchmark latent linkage package in `analysis-latent-crossarc`.
6. The cross-source efficiency writeup in `analysis-efficiency/papers/efficiency`.
7. The literature synthesis of URM, TRM, VARC, and McGovern's test-time adaptation study in `analysis-non-llm/papers/arc-synthesis`, supported by the PDFs in `papers-literature`.

Across those sources, four project-level questions recur:

1. Do humans and LLMs find the same ARC tasks easy and hard?
2. If they partly agree, what explains the agreement and what explains the divergence?
3. Do current non-LLM systems look more human-like, less human-like, or simply different?
4. What kind of computational mechanism appears to matter most for ARC success?

## 3. Methodology

Methodology needs to be laid out carefully because the project uses several different units of analysis, and many of the apparent tensions between papers are actually tensions between units, coverage, and benchmarking choices rather than outright contradictions.

### 3.1 Workstream Summary

| Workstream | Main unit of analysis | Primary inputs | Main estimators | Main limitation |
| --- | --- | --- | --- | --- |
| LLM psychometrics | Model x task binary response matrix | ARC-AGI-1 and ARC-AGI-2 public-eval prediction corpora; broader 203-model benchmark matrix | PCA, Rasch/IRT, Loevinger H, permutation nulls preserving row/column marginals | Deterministic rows, changing model catalogs, and benchmark-specific score scales |
| Human psychometrics | Session x task-pair sparse matrix, then pair/task summaries | `data-human/test_pair_attempts.csv` plus ARC task metadata | Regularized person-plus-item logistic models, split-half simulations, overlap comparisons | Sparse and biased human coverage |
| Non-LLM comparison | ARC-2 task-pair overlap; ARC-1 sidecar | TRM submissions, VARC dumps, CompressARC summaries, human overlap table | Thresholded correlations, fixed-accuracy nulls, feature controls, complementarity tests | Very small overlap and low raw accuracy |
| Solver complexity | Task-level approved solver rows | Approved Python solutions and validated task variants | Static/dynamic code metrics, PCA/PCR/PLS, permutation and bootstrap tests | Final solver size is only a proxy for latent task complexity |
| Latent cross-ARC | Task-level human latent estimates and common-model benchmark linkage | Human logs, ARC-1 and ARC-2 task JSONs, LLM matrices, complexity table | Partial pooling, split-half stability, anchored linkage via common models | ARC-1 human coverage is a sidecar, not an independent benchmark |
| Efficiency | Run/session/task summaries across source families | LLM telemetry, human session summaries, TRM/VARC/CompressARC outputs | Correlations, PCA, nested regressions, predictive models | Resource measurements are not naturally on the same scale across families |

### 3.2 Human Data and Human Difficulty Estimation

The human core is the ARC-AGI-2 log file `data-human/test_pair_attempts.csv`. The main human paper analyzes 4,681 attempts across 509 sessions, 442 task IDs, and 502 task-pair rows. Coverage is sparse: the observed session-by-item matrix has only 1.83% density, and most items have around 8 to 15 observed attempts. That sparse matrix is modeled with a regularized person-plus-item logistic regression, which serves as a practical partial-pooling approximation for ability and difficulty in the absence of a clean dense testing design.

Two methodological choices matter here. First, the project distinguishes pair-level from task-level human difficulty whenever possible. This is crucial because humans interact with individual test pairs, not with an abstract task object. Second, the project does not trust raw solve rates by default. Split-half simulations are used as a reliability reference, and later cross-ARC work shows that latent human task estimates are more stable than raw means under session split-halves.

The main caveat is sampling bias. Attempted tasks are systematically larger than unattempted tasks, especially on Public Train, so the human data are informative but not representative of the full public ARC-AGI-2 pool.

### 3.3 LLM Data and LLM Difficulty Estimation

The LLM side is built from exact-match public-eval prediction corpora. A task is scored as solved only if every test pair is solved, so the core matrices are binary model-by-task response matrices rather than soft or partial-credit tables. The original ARC psychometric paper works with a cleaner common-model panel across ARC-AGI-1 and ARC-AGI-2. Later human-facing and complexity-facing analyses also use a larger ARC-AGI-2 prediction corpus to form average-model and best-model profiles.

The main LLM psychometric estimators are PCA, Rasch-style latent ability/difficulty models, scalability diagnostics, and permutation nulls that preserve row and column marginals. That last point matters because the models are effectively deterministic test-takers: traditional residual-based psychometric p-values become misleading when there is no within-model response variance. The broader benchmark-manifold extension then asks whether ARC is an isolated oddity or part of a more general structure across coding, math, science, knowledge, long-context reasoning, instruction following, tool use, and hallucination resistance.

### 3.4 Non-LLM Comparison Design

The non-LLM paper centers the shared ARC-AGI-2 Public Eval subset because that is where human attempts and stored machine predictions overlap. The primary benchmark is the set of 110 task pairs with at least 8 human attempts, with threshold-sensitivity reruns at 2, 3, and 5 attempts as well. Human split-half correlations are used as a practical upper reference for how strongly any external profile could align with noisy human item means on that subset.

TRM evaluator submissions, VARC prediction dumps, and CompressARC summaries are then scored against the same truth files. The paper does not rely only on raw human-correlation coefficients. It also uses coarse feature controls, fixed-accuracy random-placement nulls, accuracy-matched LLM comparisons, complementarity tests on human-easy and LLM-hard items, and training-trajectory checks for TRM. That is a methodological strength of the non-LLM package: even though the top-line result is weaker than hoped, the paper asks the right null questions.

### 3.5 Solver Complexity Pipeline

The solver-complexity package begins upstream of all the psychometric comparisons by constructing the approved solver corpus. The project fetches 511 `solution.py` files from `arc.huikang.dev`, validates them against local ARC task JSONs, and then restricts analysis to the 120 approved solutions that actually pass validation. That move is critical. Without it, the complexity panel would be contaminated by incorrect or broken programs.

Static metrics include nonblank lines, tokens, AST node count, function counts, branch counts, cyclomatic complexity, nesting depth, gzip size, and Halstead measures. Dynamic metrics include opcode counts, branch opcodes, Python call counts, runtime, and memory. PCA then reveals a dominant structural factor plus weaker runtime-style axes, while PCR, PLS, and ridge models check whether supervised composites meaningfully outperform the best single metric. The complexity tables are then joined to human, LLM, and non-LLM difficulty summaries on the task overlaps that actually exist.

### 3.6 Cross-ARC Linkage and ARC-1 Sidecar Logic

The latent cross-ARC package exists because the direct human overlap with approved solver tasks is too narrow to carry the whole project. Since the workspace does not contain a standalone ARC-AGI-1 human benchmark, the analysis constructs an ARC-1 sidecar from ARC-AGI-2 Public Train responses on tasks that match ARC-AGI-1 eval task IDs and single-pair structure. That is useful, but it is not equivalent to a dedicated ARC-1 human study.

The same package also checks how ARC-AGI-1 and ARC-AGI-2 should be treated relative to one another. They can be linked via common models, but they are not interchangeable: ARC-AGI-2 is much harder, larger on average, and even the nominally shared eval task IDs use changed training examples.

### 3.7 Efficiency and Cross-Source Tradeoff Analysis

The efficiency package takes a different angle. It summarizes LLM runs, human sessions, shared ARC-2 task-level overlaps, and non-LLM records using performance, duration, cost, token, and effort variables. This package is not as central to the main causal claims as the human, LLM, and complexity packages, but it is useful for triangulating which kinds of effort move with performance inside each source family and whether shared task difficulty aligns across families at the level of score, duration, and cost.

### 3.8 Statistical Discipline

Across the project, the main inferential tools are Pearson or Spearman correlations, bootstrap confidence intervals, permutation p-values, bootstrap differences of correlations, grouped binomial models for thinking-versus-standard contrasts, OLS residualization for shared-versus-specific axis checks, and Benjamini-Hochberg false-discovery correction for named hypothesis families. This part of the project is stronger than a casual read might suggest. The pipeline is not just a stack of scatterplots. It repeatedly asks whether a claimed pattern survives when the right null is applied.

## 4. Results

### 4.1 Humans and LLMs Share a Real Difficulty Axis, But Not an Identical One

The cleanest shared-axis result is that humans and LLMs are not solving unrelated subsets of ARC. On the well-sampled ARC-AGI-2 Public Eval overlap, human difficulty and average-model difficulty correlate at about `r = 0.402`, while human split-half reliability on the same subset has a median Pearson correlation around `0.399` to `0.400`. That does not mean humans and models are "the same," but it does mean the strongest version of the randomness claim fails: the average LLM profile is clearly related to human item difficulty.

The more refined task-level comparisons point in the same direction. In the complexity package, human task difficulty and LLM task difficulty correlate around `r = 0.531`, and human solve rate aligns with average-model pass rate around `r = 0.541` on the matched approved-task overlap. The latent cross-ARC package widens the matched task coverage and still finds moderate human-versus-LLM alignment: roughly `r = 0.309` on the ARC-1 sidecar and `r = 0.355` on ARC-2 eval when latent human difficulty is compared with LLM difficulty.

The important nuance is that the most human-aligned profile is not simply the highest-scoring single model. The average-model profile repeatedly aligns with human difficulty better than the best-score single model does. That is an interpretive warning: score and human-likeness are separable in this benchmark, and ensemble-style averages can smooth away some of the idiosyncrasies of individual systems.

### 4.2 Structural Solver Complexity Is the Strongest and Most Stable Predictor of LLM Difficulty

The most surprising and most durable result in the repository is the solver-complexity result. Across the validated approved solver set, structural program metrics cluster tightly and predict LLM difficulty strongly. In the earlier latent-scale work, `ast_node_count` reaches about `r = 0.666` against latent difficulty. In the expanded LLM work, `log1p(cyclomatic_complexity)` reaches about `r = 0.707` against simple LLM logit difficulty. A supervised composite improves on the best single metric only modestly, which is itself informative: there really is one dominant structural axis here.

More importantly, the complexity signal is not equally strong for humans. On the shared approved ARC-2 overlap, cyclomatic complexity correlates only weakly with human difficulty (`r = 0.150`) but much more strongly with LLM difficulty (`r = 0.591`), and the difference-of-correlation test survives correction (`delta r = 0.441`, `q = 0.047`). The pooled benchmark-adjusted extension in the master paper keeps the same direction and strengthens the sense that this is not a fluke of one tiny slice.

This is the strongest cross-paper differentiator in the current project. The evidence does not say that humans ignore structure. It says that the burden imposed by final solver structure is much closer to the burden LLMs face than to the burden humans face.

### 4.3 Human Difficulty Looks More Time-Sensitive and More Pair-Heterogeneous

The human papers consistently show a different signature. Human difficulty tracks duration far better than raw board size. In the pair-level analyses, human pair difficulty correlates with mean human duration around `r = 0.391`, while raw input-cell count is weak or near zero. The same broader pattern appears elsewhere: human difficulty versus weighted mean duration is positive in the overlap studies, while solver structure explains relatively little by itself.

The second human-specific result is heterogeneity inside nominal tasks. Public-eval tasks with multiple test pairs show a mean within-task difficulty range of `0.719`, with a maximum of `2.498`, and task identity explains a large share of pair-level human difficulty variance. That matters because the complexity object is a task-level solver program, whereas the human behavioral burden is often experienced at the pair level. A single final solver file can therefore be an imperfect description of what makes a task hard for a person.

Taken together, the human evidence suggests that people are paying a different cost. They are not simply charged for the size or branching of the final successful program. They appear to be charged more for search, ambiguity resolution, interaction time, and per-pair variation.

### 4.4 The Non-LLM Results Are Mid, But They Are Not Empty

The disappointing headline is straightforward: on the current ARC-2 overlap, the non-LLM systems do not reproduce human difficulty nearly as well as the LLM consensus does. The LLM average reaches human-correlation values around `0.402`, while the best TRM profile reaches about `0.214` and the best VARC profile about `0.158` on the primary subset. Human-equivalence nulls are not rejected for the LLM aggregate, but they are rejected for the leading non-LLM profiles.

Still, the more careful view is not that non-LLMs are useless. First, the ranking is stable across human-attempt thresholds, which means the weak non-LLM alignment is not just a threshold artifact. Second, the non-LLM systems rescue a small but non-random set of human-easy and LLM-hard items. The TRM+VARC union rescues 6 such ARC-2 items, which is more than expected by chance in the complementarity analysis. Third, the TRM trajectory itself is interesting: human-alignment peaks at a mid-training checkpoint and then weakens as raw ARC score continues to improve. Optimization is clearly not pushing monotonically toward the human item profile.

The complexity package supports the same qualitative conclusion from another angle. On the shared 17-task ARC-2 overlap, non-LLM difficulty correlations with solver structure tend to sit between the human and LLM correlations. But those comparisons are badly underpowered. With `n = 17`, moderate effect sizes are very hard to distinguish. So the fairest summary is that the "non-LLM in the middle" pattern is descriptively real-looking but not yet statistically decisive.

### 4.5 ARC Looks Psychometrically Clean on the LLM Side, and the Broader Benchmark Manifold Mostly Agrees

The LLM psychometric paper and export-package addendum argue for a strong general factor both inside ARC and across a wider benchmark panel. On the 203-model, 13-benchmark matrix, a first component dominates, the signal is broad rather than benchmark-specific, and common score transformations preserve the loading structure almost perfectly. Richer factor structures fit better than a pure single-factor model, but they do so on top of a strong shared component rather than in place of one.

That broader result matters for the ARC-specific story because it reduces the temptation to treat ARC as a psychometric anomaly. ARC appears to be part of a broader generality manifold, although it also remains a distinctive benchmark. Hallucination resistance is the clearest outlier in the benchmark package, and expert occupational arenas show weaker generality and stronger domain-specific variance than the standard benchmark panel does.

The held-out benchmark prediction result is especially important here. If common benchmark ability predicts held-out benchmark performance well, then the general factor is doing more than re-labeling a leaderboard. It is capturing structure that generalizes across benchmark families.

### 4.6 Efficiency Results and the Small-Model Literature Point Toward the Same Mechanistic Picture

The efficiency package adds a practical cross-source perspective. Shared-task human-versus-LLM score alignment is moderate to strong, shared-task human-versus-TRM score alignment is near zero, and LLM performance tracks thinking rank, duration, and cost strongly within the LLM family. Human performance tracks latent ability strongly, but human duration remains hard to predict from geometry or machine features alone. That fits the rest of the evidence: LLM success is partly governed by machine-side computational budgets that show up cleanly in telemetry, whereas human time cost captures something less reducible to simple task geometry.

The external paper synthesis then provides a mechanistic frame for why non-LLM systems may still matter despite the current psychometric disappointment. URM, TRM, VARC, and McGovern's test-time adaptation study all converge on roughly the same computational loop: encode the task with a strong prior, iteratively refine a hypothesis, decode a candidate answer, and select among candidates using test-time adaptation or consistency checks. This maps surprisingly well onto the empirical picture from the internal analyses. The LLM side looks heavily structure-sensitive. The human side looks search-sensitive. And the non-LLM side looks most promising precisely when it leans into explicit adaptation, recurrence, or candidate selection rather than static feedforward inference.

## 5. Methodological Pressure Points

The main results are coherent, but there are several places where the methodology should be scrutinized hard before any final paper version is treated as settled.

### 5.1 Unit Mismatch Is Not a Minor Detail

Some comparisons are task-level, some are pair-level, and some move back and forth between the two. This is not cosmetic. Human burden is often realized at the pair level, while solver complexity is defined at the task level. Any argument that humans "do not care about structure" would be too strong; the safer claim is that task-level final-program structure does not explain pair-level human difficulty nearly as well as it explains LLM difficulty.

### 5.2 The Fully Matched Human + LLM + Non-LLM + Complexity Slice Is Tiny

The most integrated overlap has only 17 independent ARC-2 tasks. That is enough to detect a large effect like `LLM structure-loading > human structure-loading`, but it is not enough to sharply distinguish a visually intermediate non-LLM profile from either side. This is the single biggest reason the non-LLM story still feels underpowered.

### 5.3 Human Sampling Bias Matters for Every Cross-System Comparison

The human dataset is opportunistic. Attempted tasks are not a balanced form sampled from the benchmark. Public Eval coverage is better than nothing, but the tasks people saw are systematically different from the tasks they did not see. Any claim that directly generalizes the human item profile to "ARC as a whole" needs to be qualified.

### 5.4 ARC-AGI-1 and ARC-AGI-2 Can Be Linked, But Not Collapsed

The cross-ARC package makes a useful contribution by anchoring the two benchmarks through common models and by building the ARC-1 human sidecar. But the linkage is not a license to pretend the benchmarks are the same test on a common ruler. ARC-AGI-2 is harder, larger, and revised in ways that matter, including changed training examples on the shared eval IDs.

### 5.5 The Thinking-Advantage Result Is No Longer a Headline Finding

One of the flashiest early results was that thinking-model advantage declines as item difficulty rises. After model-label auditing and floor-sensitive checks, that claim weakened substantially. It should remain in the draft only as a downgraded exploratory thread or as evidence that the pipeline is capable of rejecting an exciting result once the taxonomy is cleaned up.

## 6. Discussion

The best integrated interpretation is not "LLMs are like humans" and not "LLMs are nothing like humans." It is that ARC exposes at least two layers of structure.

The first layer is shared. Humans, the LLM average, and even some non-LLM systems are not solving arbitrary disjoint subsets of tasks. There is a common difficulty axis that shows up across human solve rates, LLM pass rates, latent human difficulty, and latent LLM difficulty. This shared layer is exactly why ARC remains interesting as a comparative benchmark.

The second layer is source-specific burden. For LLMs, final solver structure appears to be load-bearing. Complexity metrics derived from validated approved programs repeatedly forecast which tasks are hard for models. For humans, that burden looks weaker, and the evidence keeps pointing toward time cost, search, and pair-level ambiguity as the more distinctive human burden. For non-LLM systems, the story is not yet one of broad human-likeness. The better reading is that these systems sometimes inject orthogonal search or inductive-bias advantages without yet recreating the human difficulty profile at scale.

This also helps explain why the small-model literature and the empirical repo analyses do not conflict. The external papers suggest that performance gains come from strong priors, iterative refinement, and test-time candidate selection. The internal analyses suggest that these ingredients can matter without automatically producing human-like psychometrics. A system can become better at ARC by exploiting the right search loop while still remaining relatively unlike humans in which items it finds easy or hard.

One implication is especially important for future writing. If the paper is framed as a humans-versus-LLMs contest, it will miss the most interesting result. The stronger contribution is a decomposition claim: ARC difficulty is partly shared, but the residual burdens are not the same. That is a more defensible and more useful conclusion.

## 7. Priority Follow-Ups

If this draft is turned into a next-pass paper, the highest-value follow-ups look fairly clear.

1. Increase the fully matched overlap. The most important unresolved question is still the non-LLM one, and that question will not be settled cleanly on 17 tasks.
2. Make the task-level versus pair-level distinction explicit in every methods and results table. A lot of confusion disappears once the unit mismatch is acknowledged directly.
3. Keep the complexity story central. It is the strongest result in the repo and the one most likely to survive scrutiny.
4. Reframe the non-LLM section around complementarity rather than around "beating LLMs at human-likeness." That is where the current data are actually interesting.
5. Treat the ARC-1 sidecar as a useful auxiliary bridge, not as independent confirmation on its own.
6. Keep the broader benchmark-manifold result in the draft, because it makes the ARC conclusions feel less isolated and more psychometrically grounded.

## 8. Conclusion

The repo now supports a fairly clear first-draft conclusion. ARC is neither a pure human test nor a pure machine benchmark. It contains a real shared abstraction factor across humans and models, but that shared factor is not the whole story. The strongest evidence in the project says that validated solver structure is a strong predictor of LLM difficulty and a much weaker predictor of human difficulty. Human difficulty looks more time- and search-sensitive, with substantial heterogeneity across test pairs inside the same nominal task. Current non-LLM systems are not yet the human-like alternative one might have hoped for, but they are not empty either: they contribute complementary successes and point toward a mechanistic loop built from inductive bias, iterative refinement, and test-time selection. The right next step is not to claim that one paradigm has won. It is to sharpen the decomposition, expand the matched overlap, and write the final paper around what the current evidence actually supports.
