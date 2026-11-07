# AMC12 hard-problem phase

当前目标：先把 AMC 后半卷题目交给更强的数学模型作答，检查能否在合理生成预算内完成；通过验证后才做三种选题策略的训练对比。历史算术和 GSM8K 是流程记录，不能证明竞赛数学推理提升。

## Exactly which AMC data?

Downloaded: [AI-MO/aimo-validation-amc](https://huggingface.co/datasets/AI-MO/aimo-validation-amc), revision `69d78a4a2c840e82d69af6bc742bda09005f6316`. The publisher documents 83 adapted AMC12 problems from 2022 and 2023, sourced from AoPS. These are **modified integer-answer variants**, not original multiple-choice AMC papers, not AMC10 and not AIME. The source card inaccurately describes original AMC choices as four; the source's variant/provenance claims should not be treated as independent verification of official exam data.

The card declares Apache-2.0, but underlying AMC redistribution rights have not been independently established. This repository publishes acquisition code, hashes, source URLs, identifiers and numerical measurements only. Problem text and full generated responses stay under ignored `work/amc/`. The adapted answer key comes from the dataset publisher; no worked solutions are present, no teacher solutions are invented, and the key has not been independently checked against every original problem.

| Source | Downloaded | Eligible positions 16–25 | Role |
|---|---:|---:|---|
| 2022 AMC12A | 22 | 9 | Training candidates |
| 2022 AMC12B | 21 | 9 | Validation only |
| 2023 AMC12A / B | 22 / 18 | 13 combined | Reserved test pool |

Position 16–25 is an explicit difficulty proxy; no easier first-half questions enter this phase. It does not guarantee that every included problem is equally hard or requires the same reasoning depth. Exact whitespace-normalized hashes are unique across all 83 source rows. Semantic duplication and model pretraining overlap remain unknown. Year/form splits prevent validation/test questions from being training examples. Final test subset IDs are not frozen or evaluated until model/generation feasibility is established. No test correctness is used for calibration.

## Student and bounded calibration

[Qwen2.5-Math-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B-Instruct), Apache-2.0 model card, revision `aafeb0fc6f22cbf0eaeed126eff8be45b0360a35`. This is a math-specialized 1.5B model, replacing the 135M student for this phase. Actual local resources: 24 GiB memory, 10 reported CPU cores, PyTorch MPS unavailable. Use float32 CPU with four Torch threads; no paid services.

Calibration takes the first three eligible 2022B validation problems sorted by normalized question hash, before seeing model outputs. Chat template with the published style of step-by-step reasoning and boxed final answers; greedy decoding; 2,048-token ceiling and a separate **90-second wall-clock limit per problem**. The 2,048 ceiling is a candidate adequate generation budget; reaching the time/token limit means it has not been validated as adequate locally. Save all continuations locally, and publish numeric parsed predictions, source IDs, correctness, EOS termination and measured speed. Pure numeric boxed/marked answers are scored exactly with Decimal; incomplete calculations and incidental numbers are rejected. These adapted problems require integer answers, so general symbolic equivalence is not claimed.

Feasibility gate fixed before calibration: at least two parseable EOS completions and one correct answer out of three, with no wall-clock limit reached. This tiny screen checks feasibility, not a benchmark score or statistically reliable model quality. On failure, do not freeze a final test set or present a curriculum comparison as completed.

## Reproduce acquisition and calibration

```bash
pip install -r requirements-amc.txt
HF_HOME="$PWD/work/hf" python amc.py --phase prepare
HF_HOME="$PWD/work/hf" python amc.py --phase calibrate
python -m unittest test_amc -v
```

## Conditional next study (not run)

Before training: enlarge the hard AMC training pool from additional older year/forms with documented usage rights and trusted solution/answer provenance. Nine answer-only training items cannot support a strong reasoning-training study. Keep 2022B validation and 2023 test untouched by training. Do not manufacture rationale labels. Full worked-solution supervision requires separately sourced verified solutions; a final-answer-only experiment must be clearly named as such.

Proposed compute: an existing user-provided CUDA GPU with at least 24GB VRAM, or explicit budget approval for a single rented GPU pilot. Use Qwen2.5-Math-1.5B-Instruct LoRA, batch one with accumulation, complete targets, no truncation, and 2,048 generation tokens initially. Measure one training and one generation batch before committing to a larger run; duration and price cannot be asserted without that measurement/provider quote. No GPU time is booked or purchased.

Comparison design: same base adapter state and seeds; same supervised-token-length schedule across random, fixed original-position curriculum and student-loss-based adaptive selection. Charge all candidate scoring separately. Evaluate base and checkpoints on the frozen held-out forms; report final numeric accuracy, EOS/completion rates, actual token counts and training/selection/evaluation cost. A transfer claim requires a separately defined held-out family or later-year test with labels/provenance; year separation alone is not proof of unseen mathematical families. This is a concrete follow-up proposal, not implemented training or measured gains.
