# ARC-1 GPT OSS + Gemini Analysis

## Current LLM Pool

- gpt-4o
- claude-3.5-sonnet
- nvidia/nemotron-3-super-120b-a12b:free
- gemma-4-31b-it
- gemini-3.1-flash-lite-preview
- gpt-5.4-nano-2026-03-17
- openai/gpt-oss-120b:free
- IceCuber remains separate and is not pooled with the LLMs.

## Coverage

- Gemini 3.1 validated coverage: 391 tasks, solve rate = 0.384.
- GPT-5.4 Nano validated coverage: 391 tasks, solve rate = 0.325.
- GPT OSS 120B validated coverage: 391 tasks, solve rate = 0.588.

## Complexity PC1

- Human: r = 0.125
- LLM pool without GPT OSS: r = 0.523
- LLM pool with GPT OSS: r = 0.556
- GPT-5.4 Nano alone: r = 0.440
- GPT OSS alone: r = 0.440

## GPT OSS Effect

- full_set: llm6_pair_difficulty r = 0.523, llm7_pair_difficulty r = 0.556, Williams p = 0.002945.
- full_set: human_difficulty_complete r = 0.125, llm7_pair_difficulty r = 0.556, Williams p = 4.249e-16.
- gap_le_0.30: llm6_pair_difficulty r = 0.431, llm7_pair_difficulty r = 0.469, Williams p = 0.02436.
- gap_le_0.30: human_difficulty_complete r = 0.352, llm7_pair_difficulty r = 0.469, Williams p = 0.02357.
