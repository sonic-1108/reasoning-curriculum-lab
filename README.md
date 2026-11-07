# What should a reasoning model learn next?

## 现在先做 AMC / AMC first

**主线已改为 AMC12 后半卷竞赛题；旧算术和 GSM8K 仅保留为历史流程验证。**

已获取 AI-MO 发布的 2022/2023 AMC12 **83 道整数答案改写题**，不是原版选择题，也不是 AIME。只采用原题号 **16–25**：2022A 训练候选 9 道、2022B 验证 9 道、2023 年保留测试 13 道。来源只有数值答案，没有标准推导；题文留在本地，GitHub 只发布代码、来源和测量记录。[准确来源、版权范围、模型和计划](AMC_STUDY.md)。

模型改为 **Qwen2.5-Math-1.5B-Instruct**，已实际作答 3 道 2022B 验证题：最多 2,048 个生成 token、每题 90 秒。**结果为 0/3；三题均正常完成，没有被截断。尚未进行 AMC 微调、三策略比较或 2023 测试。**

| 2022 AMC12B 改写题 | 模型答案 | 数据集答案 | 生成 token | 耗时 |
|---|---:|---:|---:|---:|
| 第 25 题 | 8 | 4 | 558 | 72.22 秒 |
| 第 17 题 | 90 | 76 | 501 | 59.02 秒 |
| 第 18 题 | 112 | 22 | 496 | 57.44 秒 |

三题是按验证题哈希预先选定的，不是挑出来的失败案例。当前 CPU 能运行这个模型，速度为 **7.73–8.64 token/秒**；瓶颈首先是本次难题正确率，而非输出上限。三个样本不足以推断整个 AMC 的成绩。由于未达到预先设置的能力门槛，测试题继续保留，暂不盲目微调这个学生。[数值预测和运行设置](amc_results/calibration.json) · [完整测量](amc_results/RESULTS.md)。

![AMC validation completion and CPU speed](amc_results/calibration.svg)

下一步需要校准更大的数学模型，并取得更多有可信解答的 AMC 训练题。当前 9 道训练候选只有数值标签，不足以支持有说服力的推理训练研究。GPU/量化模型方案是待执行计划，不是已有结果；未购买任何计算资源。

```mermaid
flowchart LR
    A[AMC12 positions 16–25] --> B[2022A training candidates]
    A --> C[2022B validation calibration]
    C --> D[Check math model and generation budget]
    D --> E[If viable: freeze later-year test]
    B --> F[Future three-policy training study]
    E --> F
```

这里的“喂题”先指让预训练模型作答；校准不更新权重。未解决模型能力、算力及可信训练解答来源前，不盲目开始微调。

## 历史 GSM8K 流程实验 / Historical GSM8K pilot

**GSM8K 固定子集实验已完成：9 次真实训练、3 个种子、456 条保存的评测回答。**

研究问题：同样的模型、同样的训练 token 预算，下一道题应该随机选、先易后难选，还是根据学生当前的错误选？模型是 **SmolLM2-135M**；策略只是选题规则。模型生成答案，训练程序用训练集标准解答更新权重，测试集只负责评分。

```mermaid
flowchart LR
    A[GSM8K training pool] --> B[Random / Easy-to-hard / Adaptive]
    B --> C[Same pretrained student]
    C --> D[Update weights on training solutions]
    D --> B
    E[Fixed held-out test questions] --> F[Student generates answers]
    D --> F
    F --> G[Final-answer accuracy and cost]
```

GSM8K has 7,473 official training and 1,319 official test problems. This CPU study used a fixed **24-question test subset**, a **360-question retained training pool**, 12 updates and three seeds, with complete worked-solution supervision. Both supervised token counts and padded training slots match across policies within each seed. [Exact experiment definition and provenance](GSM8K_METHOD.md).

| GSM8K subset | Final accuracy | Correct / 24 by seed | Train seconds | Select seconds |
|---|---:|---|---:|---:|
| Untrained base | 0.0% | 0 (evaluated once) | — | — |
| Random | 0.0% | 0, 0, 0 | 6.82 | 0.0002 |
| Easy-to-hard | 1.4% | 0, 0, 1 | 6.83 | 0.0002 |
| Adaptive | 2.8% | 1, 1, 0 | 6.88 | 6.41 |

**结论：当前小模型、短训练和生成预算下，没有得到可信的应用题能力提升。** 自适应比随机多得到两个数字匹配，但训练加选题耗时约 **1.95 倍**，不能据此宣布胜出。最终回答有 **95.8%–97.2%** 达到 96-token 上限；部分匹配数字伴随错误推导。这里测的是最终数字匹配，不是推理正确性，也不是完整 GSM8K 成绩。

The same 24 questions are reused across seeds; 72 final observations are not 72 independent questions. Token budgets per seed are **958 / 1,165 / 1,221 supervised tokens**, identical across policies; every run also uses **3,072 padded slots**. Times are per-run seed means, excluding evaluation. [Full measured table](gsm8k_results/RESULTS.md), [raw predictions](gsm8k_results/predictions.json), [selection logs](gsm8k_results/selections.json), [evaluation timing](gsm8k_results/costs.json), [provenance and fixed IDs](gsm8k_results/manifest.json).

![GSM8K answer and format curves](gsm8k_results/figures/supervised_tokens.svg)
![GSM8K accuracy versus training and scoring cost](gsm8k_results/figures/cost.svg)

曲线中的答案格式比例用于解释模型是否输出可评分答案，不表示推理能力。All charts are regenerated from saved predictions, using the corrected strict final-answer parser. Five tests pass; the benchmark audit checks all 456 predictions, complete test checkpoints, training-only selections, and exact matched token schedules. Unknown pretraining overlap with GSM8K and semantic duplicates cannot be excluded.

```bash
HF_HOME="$PWD/work/hf" python gsm8k.py
python -m unittest -v
python gsm8k_report.py
```

Next scientific step: select a more capable student and a longer generation/training budget using a validation split carved from training data, then freeze a new experiment before evaluating. This run used CPU only and incurred no paid cloud charges. It demonstrates a working standard-dataset experiment and its limitations, rather than a successful curriculum method.

## Earlier arithmetic pilot

**Status: completed — 9 real open-model training runs, 3 seeds, saved predictions and reproducible figures.**

If a model has time to study only a few examples, which examples should it study?
This project compares random examples, examples ordered from easy to hard, and examples chosen using the student's current answer loss.

We fine-tuned **SmolLM2-135M**, a public 135-million-parameter language model, on CPU for 30 updates per policy with seeds 7, 19 and 41. Adaptive selection had the highest average answer accuracy, but used almost twice the training-plus-selection time. **None of the policies solved a held-out composition problem.** This small arithmetic experiment offers no evidence of improved transfer to new reasoning families.

| Policy | Mean accuracy | Correct by seed (out of 64) | Training seconds | Selection seconds |
|---|---:|---|---:|---:|
| Random | 17.7% | 12, 9, 13 | 16.64 | 0.001 |
| Easy-to-hard | 16.7% | 12, 11, 9 | 16.81 | 0.001 |
| Adaptive | 20.8% | 13, 13, 14 | 17.21 | 15.66 |

Times are means over three runs. All start at **1/64 (1.6%)**. Evaluation uses the same 64 problems for every run: 19 held-out questions from seen families and 45 unseen composition questions. Final seen-family accuracy is 59.6%, 56.1%, and 70.2% respectively; composition accuracy is **0% for all three**. The 3.1 percentage-point adaptive gain over random is small and measured on one fixed test set; no statistical superiority claim is made.

![Learning curve](results_full/figures/slots.svg)
![Compute cost](results_full/figures/cost.svg)
![Families](results_full/figures/family.svg)
![Difficulty](results_full/figures/difficulty.svg)
![Transfer](results_full/figures/split.svg)

Bold curves show seed means; faint curves show individual runs. All figures derive from [saved predictions](results_full/metrics.json), with [summary measurements](results_full/summary.json), [selection logs](results_full/selections.json) and [environment details](results_full/environment.json). The earlier 10-step single-seed pilot remains in `results/` (random 14.1%, easy-to-hard 12.5%, adaptive 15.6%).

## What actually happened

1. Generate 800 unique arithmetic questions with exact answers: addition, subtraction, multiplication, and a held-out composition family. Save every question in `results_full/data.json`: 485 training, 115 seen-family held-out and 200 composition questions.
2. Split seen families by SHA-256 of the prompt (roughly 80% training). Reserve every composition problem for evaluation. Select a fixed 64-question evaluation sample before training.
3. Reload identical pretrained weights and reset the random seed for each policy. Use AdamW, learning rate 0.00002, batch size 8, and padded length 32.
4. Random: sample 32 training candidates and use eight. Easy-to-hard: choose eight candidates nearest the scheduled difficulty. Adaptive: score 32 training candidates and prefer losses nearest 1.5 nats, a fixed heuristic for examples the student may learn from. Evaluation answers never influence selection.
5. Train on answer tokens only. Evaluate greedy generation with strict exact answer matching at step 0 and every 10 updates. Save individual predictions and selected example IDs.
6. Plot saved measurements using `report.py`. Selection/scoring time is added to training time in the cost curve. Generation and evaluation are recorded separately in environment metadata.

Each final run uses **7,680 padded token slots**, including padding and prompt positions. Actual supervised answer-token means are 914.7 (random), 960.0 (easy-to-hard) and 900.0 (adaptive); this is equal tensor-compute budget, not exact equal non-padding token count. Timing is local elapsed time, not dollars or energy. Adaptive's training-plus-selection cost is 32.87 seconds versus 16.64 for random (**1.98×**). Evaluation and setup are excluded from this cost comparison.

## Run it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
python -m unittest -v
python validate_results.py results
python experiment.py --steps 10 --seeds 7 --output results
python report.py results
# Larger replication (more time):
python experiment.py --steps 30 --seeds 7 19 41 --output results_full
python report.py results_full
python validate_results.py results_full
```

The model downloads from Hugging Face; no paid APIs are used. CPU fallback and Apple MPS are supported; the measured runs used CPU on macOS ARM64 with four Torch threads. `environment.json` records the model revision, device, framework, configuration and evaluation IDs. `requirements-lock.txt` captures the measured environment. The model revision is pinned by default; override `--revision` when using a different `--model`. See [METHODS.md](METHODS.md) for the full experimental record.

## How much does this tell us?

This is an arithmetic engineering pilot, with actual pretrained-model fine-tuning. It does not measure advanced mathematical reasoning. Thirty updates, three seeds, 64 questions and a very small model give limited statistical power. Strict formatting may reject numerically correct continuations. The unseen family uses familiar operations; it is a narrow transfer test. Prompt deduplication prevents exact split overlap; semantically related arithmetic and overlap with unknown pretraining data remain possible. Difficulty is a declared operand/operator heuristic, not a validated human rating. Full-model checkpoints are not uploaded; rerun commands reproduce training. Two data tests and a saved-run audit verify arithmetic answers, stable splits, unique IDs, identical initial predictions, complete checkpoints, equal padded budgets and training-only selections. The final audit covers **2,304 saved predictions**.

## Research context

[DART-Math](https://arxiv.org/abs/2407.13690) allocates more synthesis trials to difficult problems. This project instead studies selection from a fixed arithmetic pool and charges for student scoring. It is a small reproducible comparison, with no novelty claim. [SmolLM2 model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) documents the public model and its Apache-2.0 license. Generated arithmetic data and code here use the MIT license.
