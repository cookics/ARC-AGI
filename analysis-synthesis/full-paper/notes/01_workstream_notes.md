# Workstream Notes For The Full ARC Master Paper

## 1. Human Data And Human Psychometrics

- Source file: `data-human/test_pair_attempts.csv`
- 4,681 human attempts
- 509 sessions
- 442 task IDs
- 502 task-pair rows
- 40.4% coverage of all ARC-AGI-2 public task pairs
- observed session-by-item matrix density: 1.83%
- overall human solve rate: 75.3%
- solve rate without warm-up: 73.4%
- human vs average-model difficulty correlation on the robust Public Eval overlap: about 0.402
- human split-half median on that overlap: about 0.399 to 0.400
- strong caution: sparse, opportunistic, and biased sampling

## 2. LLM Psychometrics

- response matrices built from exact-match public-eval predictions
- ARC behaves strongly one-factor dominated on the LLM side
- broader 203-model benchmark matrix also shows a strong common component
- held-out benchmark prediction is one of the strongest addendum results
- deterministic-test-taker framing is methodologically essential
- richer factor models can fit better without displacing the shared factor

## 3. Non-LLM Psychometrics

- primary ARC-2 subset: 110 Public Eval pairs with at least 8 human attempts
- ARC-2 eval overlap table also considers 161 pairs before the threshold
- ARC-1 sidecar: 230 single-pair ARC-1 tasks reused from ARC-AGI-2 Public Train
- LLM average human-correlation: about 0.402
- best-aligned LLM: about 0.439
- best-aligned TRM: about 0.214
- best VARC: about 0.158
- TRM+VARC union rescues 6 human-easy / LLM-hard ARC-2 items
- strongest conclusion: non-LLMs are weaker than the LLM consensus at reproducing human difficulty, but not redundant

## 4. Solver Complexity

- 511 fetched Python solutions
- 127 passed validation
- 120 approved solutions passed out of 120 approved
- approved-only package: 120 tasks
- strongest LLM-side structural predictor: log1p cyclomatic complexity around 0.707
- earlier strong single metric: AST node count around 0.666
- human task difficulty vs LLM task difficulty on approved overlap: about 0.531
- cyclomatic complexity vs human difficulty: about 0.150
- cyclomatic complexity vs LLM difficulty: about 0.591
- difference survives corrected testing and is the strongest repo result
- residual LLM difficulty still tracks structure after removing shared human difficulty
- human burden is more duration- and pair-heterogeneity-sensitive

## 5. Cross-ARC Latent Linkage

- human benchmark slices: `arc1_sidecar`, `arc2_eval`, `arc2_train_other`
- latent task estimates are more stable than raw solve rates in all three slices
- common-model ARC-1 vs ARC-2 accuracy correlation is very high
- 6 shared eval IDs preserve test examples but not training examples
- ARC-1 sidecar matched human/LLM tasks: 230
- ARC-2 eval matched human/LLM tasks: 115
- pooled structure story still favors stronger LLM structure-loading than human structure-loading

## 6. Efficiency Analysis

- 29 LLM model-run rows
- 509 human session rows
- 15 non-LLM run rows
- 115 shared ARC-2 task rows
- shared-task human vs LLM score alignment is moderate to strong
- shared-task human vs TRM score alignment is near zero
- LLM performance tracks thinking rank, duration, and cost strongly
- human performance tracks latent ability strongly
- geometry plus LLM performance helps predict human solve rate more than geometry alone
- human duration remains poorly explained

## 7. External Architectural Synthesis

- URM, TRM, VARC, and McGovern all converge on a loop of:
  strong inductive bias,
  iterative refinement or adaptation,
  and test-time candidate selection
- this external line helps explain why non-LLM systems can be architecturally important even when their current human-alignment numbers are modest

## 8. Master Through-Line

The strongest integrated repo thesis is:

- there is a real shared difficulty axis across humans and machines,
- solver structure is a strong predictor of LLM difficulty,
- human burden is less reducible to final solver structure and more tied to duration, search, and pair-level heterogeneity,
- and current non-LLM systems add complementary signal without yet beating the LLM consensus on human-likeness.
