# Thinking Label Sources

These sources were used to resolve the ambiguous model labels in [thinking_label_audit.csv](C:/Users/cooki/Desktop/ARC-AGI/Python%20solutions/approved_only/thinking_label_audit.csv).

- OpenAI pro reasoning support:
  [GPT-5.4 pro model docs](https://developers.openai.com/api/docs/models/gpt-5.4-pro)
  The docs state that GPT-5.4 pro supports `reasoning.effort: medium, high, xhigh`.
  This was used as provider-side evidence that `gpt-5-pro-*` and `gpt-5-2-pro-*-high/medium` are reasoning-style variants rather than plain standard models.

- Google Gemini Pro thinking:
  [Gemini thinking docs](https://ai.google.dev/gemini-api/docs/thinking)
  The docs state that you cannot disable thinking for Gemini 3 Pro.
  This was used to classify `gemini-3-pro-preview` as a Thinking model.

- Alibaba QwQ reasoning:
  [QwQ-32B deployment docs](https://www.alibabacloud.com/help/doc-detail/2874902.html)
  [QwQ-32B announcement](https://www.alibabacloud.com/blog/602039)
  These describe QwQ-32B as a reasoning/inference model.
  This was used to classify `QwQ-32B-Fireworks` as a Thinking model.

## Caveat

Not every historical snapshot in the local dataset still has a directly accessible archived vendor page. For the dated OpenAI `gpt-5-2-pro-*` snapshots, the classification relies on the provider's current pro-model documentation plus the local snapshot naming convention (`high`, `medium`) rather than an archived page for that exact snapshot.
