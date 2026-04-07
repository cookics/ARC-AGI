# Audit Deck Summary

## Strongest Similarity Signals

- Human difficulty vs LLM logit difficulty: `r = 0.531` on `n = 17`.
- Human solve rate vs all-model pass rate: `r = 0.541` on `n = 17`.
- Human solve rate vs thinking-model pass rate: `r = 0.551` on `n = 17`.
- LLM pooled Rasch vs LLM logit difficulty: `r = 0.970` on `n = 55`.
- LLM previous latent vs pooled Rasch: `r = 0.991` on `n = 55`.

## Strongest Difference Signals

- Cyclomatic complexity: human difficulty `r = 0.150` vs LLM difficulty `r = 0.591`.
- Human duration: human difficulty `r = 0.425` vs LLM difficulty `r = 0.050`.
- Thinking advantage vs difficulty: `r = -0.492` on `n = 56`.
- Residual human difficulty after removing LLM difficulty still tracks duration: `r = 0.470`.
- Residual LLM difficulty after removing human difficulty still tracks cyclomatic complexity: `r = 0.603`.
- Human within-task difficulty range across test pairs: mean `0.719`, max `2.498` across `44` tasks.

## Charts

- `chart_audit_shared_signals.png`
- `chart_audit_divergence_signals.png`
- `chart_audit_human_specific_signals.png`