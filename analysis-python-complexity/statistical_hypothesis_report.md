# Statistical Hypothesis Audit

## Null-Hypothesis Framing

- For correlations, the null is exchangeability: shuffling task or pair identities should not yield a stronger association than observed.
- For difference-of-correlation claims, the null is equal association with the two outcomes on the same sampled rows.
- For `thinking_advantage`, the null is that model group does not interact with item difficulty in a grouped binomial-logit model.

## Thinking-Advantage Derivation

- `thinking_advantage = pass_rate_thinking - pass_rate_standard`.
- `thinking_logit_advantage` uses the same group counts on a smoothed log-odds scale.
- Labels are audited in [thinking_label_audit.csv](C:/Users/cooki/Desktop/ARC-AGI/Python%20solutions/approved_only/thinking_label_audit.csv).

## Ambiguous Models

- QwQ-32B-Fireworks, gemini-3-pro-preview, gpt-5-2-pro-2025-12-11-high, gpt-5-2-pro-2025-12-11-medium, gpt-5-pro-2025-10-06

## Key Tests

- `S1` Human and LLM task difficulty are positively aligned on the approved ARC-2 eval overlap. Estimate `0.531`, CI `[0.178, 0.780]`, p `0.02683`, q `0.03659`.
- `S2` Human solve rate aligns with average-model pass rate on the same overlap tasks. Estimate `0.541`, CI `[0.226, 0.781]`, p `0.01867`, q `0.028`.
- `S3` The older shared-model latent scale and the new pooled Rasch scale are effectively the same LLM difficulty axis. Estimate `0.991`, CI `[0.984, 0.997]`, p `0.0001667`, q `0.0003571`.
- `S4` Pooled Rasch difficulty and simple LLM logit difficulty are almost identical on the approved subset. Estimate `0.970`, CI `[0.953, 0.981]`, p `0.0001667`, q `0.0003571`.
- `D1` Cyclomatic complexity is more strongly associated with LLM difficulty than with human difficulty. Estimate `0.441`, CI `[0.024, 0.909]`, p `0.03775`, q `0.04719`.
- `D2` Human duration is more strongly associated with human difficulty than with LLM difficulty. Estimate `0.374`, CI `[-0.058, 0.803]`, p `0.082`, q `0.08786`.
- `D3` Human-specific residual difficulty still tracks human duration after removing shared LLM difficulty. Estimate `0.470`, CI `[0.067, 0.766]`, p `0.0505`, q `0.05826`.
- `D4` LLM-specific residual difficulty still tracks solver structure after removing shared human difficulty. Estimate `0.603`, CI `[0.284, 0.816]`, p `0.01167`, q `0.01944`.
- `H1` On well-sampled public-eval pairs, human difficulty tracks human time cost. Estimate `0.391`, CI `[0.238, 0.539]`, p `0.0001667`, q `0.0003571`.
- `H2` On well-sampled public-eval pairs, raw board size alone is weak for human difficulty. Estimate `-0.096`, CI `[-0.275, 0.086]`, p `0.3108`, q `0.3108`.
- `H3` Human duration is more strongly associated with human difficulty than raw board size is. Estimate `0.487`, CI `[0.255, 0.713]`, p `0.000125`, q `0.0003571`.
- `H4` Human-over-LLM advantage shrinks when public-eval tasks expose more test pairs. Estimate `-0.366`, CI `[-0.539, -0.165]`, p `0.0003333`, q `0.0006249`.
- `H5` Task identity explains a large share of pair-level human difficulty variation on multi-pair tasks. Estimate `0.749`, CI `n/a`, p `7.934e-05`, q `0.0003571`.
- `T1` Thinking advantage declines as approved-item LLM difficulty rises under the legacy label schema. Estimate `-0.492`, CI `[-0.813, -0.165]`, p `0.0001667`, q `0.0003571`.
- `T2` Thinking-vs-standard success probability has a negative difficulty interaction under the legacy label schema. Estimate `-0.718`, CI `[-1.108, -0.358]`, p `5.697e-06`, q `8.545e-05`.

## Thinking Sensitivity

- Schema `legacy`: raw-gap `r = -0.492`, p `0.0001667`; logit-gap `r = -0.510`, p `0.0003333`; GLM interaction `-0.718`, Wald p `5.697e-06`, both-zero items `0`.
- Schema `strict`: raw-gap `r = -0.538`, p `0.0001667`; logit-gap `r = -0.595`, p `0.0001667`; GLM interaction `-0.784`, Wald p `2.841e-08`, both-zero items `0`.
- Schema `maximal`: raw-gap `r = -0.378`, p `0.0045`; logit-gap `r = -0.294`, p `0.0245`; GLM interaction `0.767`, Wald p `0.009033`, both-zero items `0`.