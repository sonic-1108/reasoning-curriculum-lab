# GSM8K: frozen local subset study

研究问题：在相同初始模型、相同训练 token 数下，根据学生当前错误选择题目，是否比随机选题或先易后难更有效？

谁在作答？SmolLM2-135M 预训练语言模型。训练程序用训练题的标准解答更新模型权重。测试只让模型作答并评分，不更新权重。“策略”是选训练题的规则，不是另一个聊天机器人。

## Configuration fixed before the local run

- Source: OpenAI's [GSM8K repository](https://github.com/openai/grade-school-math), pinned commit `3101c7d5072418e28b9008a6636bde82a006892c`; [dataset card](https://huggingface.co/datasets/openai/gsm8k) specifies MIT license. Upstream license is retained in `gsm8k_results/DATA_LICENSE` when results are published.
- Official splits: 7,473 training and 1,319 test word problems. Fixed local evaluation: 24 official test examples selected with seed 2027 before any model evaluation. These are subset scores, not full GSM8K scores.
- Training pool: first 512 eligible examples from a shuffle of official training data with seed 2026. Filter complete question+solution sequences to at most 256 tokens. Remove normalized exact question duplicates and any exact official-test overlap. Keep target-length groups with at least four examples. Record actual retained pool size and original test indices in manifest.
- Student: `HuggingFaceTB/SmolLM2-135M`, model revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, full-weight CPU fine-tuning, four Torch threads, AdamW 2e-5, batch size one, 12 updates, seeds 7/19/41.
- Equal budget: before training, draw a target-token-length schedule using only training-data lengths and the run seed. All policies follow the same schedule. At each update select among four training candidates with exactly that target length. Thus both total supervised tokens and padded tensor slots match exactly across policies within each seed. This length constraint limits selection freedom and is a substantive limitation.
- Random selects the first randomly ordered candidate. Easy-to-hard chooses solution calculation-count nearest a fixed linearly increasing target from 1 to 6. Calculation count is a proxy, not validated difficulty. Adaptive scores only training candidates and chooses loss closest to the fixed 1.5-nat target. This remains a heuristic rather than an estimated causal learnability measure.
- Supervision: full published worked solution plus final answer, including EOS. Tokenize prompt and target separately; concatenate without truncation; mask every prompt and padding label. Assert the complete target is supervised. The question-only prompt contains no solution.
- Evaluation: base weights once, then checkpoints at updates 6 and 12. Greedy decoding, at most 96 new tokens. Parse `####`, boxed answers, explicit final-answer statements, or bare numeric responses. Normalize with Decimal, including comma separators. Reject incidental numbers in unfinished explanations. Save raw generations, parsed answers, token lengths and correctness. Generation truncation can lower observed accuracy; report token-cap frequency.
- Costs: measure selection/scoring, training and evaluation separately with elapsed CPU time. Setup/download is separate. No paid compute, teacher generation or test-guided tuning.

## Interpretation limits

The 135M student and tiny training budget are a local feasibility study. A result of zero would be a measured failure at this configuration, not evidence that curriculum learning cannot work. Evaluation examples are shared across seeds and cannot be treated as independent samples for significance claims. GSM8K lacks official problem-family labels: this phase measures held-out word problems, not unseen-family generalization. The earlier arithmetic composition experiment remains the only explicit unseen-family test.

Unknown pretrained-model overlap with GSM8K cannot be excluded. Normalized question checks prevent exact pipeline overlap but cannot prove semantic decontamination. Filtering training sequences to 256 tokens changes the training distribution; short solutions are overrepresented. No claim of benchmark novelty or full-benchmark improvement is intended.

Scoring audit: during inspection of raw generations, malformed calculation lines such as `#### 48/8 = 6` and percentage ratings were found to be incorrectly accepted by a permissive prefix parser. The final parser requires a complete numeric marker line, with regression tests. Every saved prediction is rescored by `gsm8k_report.py`; live console counts before this fix are provisional. This parser correction changes no model weights, selection rules, budgets or decoding settings.

## Reproduce

```bash
HF_HOME="$PWD/work/hf" python gsm8k.py
python -m unittest -v
python gsm8k_report.py
```

Do not use evaluation results to change this configuration retrospectively. Any follow-up configuration should be separately named and motivated, with a fresh validation split from training for tuning.

## Completed outcome

All nine configured training runs finished. Final counts across seeds 7/19/41 were random 0/0/0, easy-to-hard 0/0/1, adaptive 1/1/0 out of 24 shared test questions. The base model scored 0/24. Exact supervised budgets were 958, 1,165 and 1,221 tokens per seed for every policy. All runs used 3,072 padded positions. The saved-run audit passed for 456 predictions and training-only selection logs. See `gsm8k_results/RESULTS.md` for timing and generation-limit measurements. These results establish local execution, not a reliable curriculum advantage or full-benchmark result.
