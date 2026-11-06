# What should a reasoning model learn next?

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
