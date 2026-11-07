# Cross-model validation of the training method

**Status: proposed design and runnable inference-screening support. The hard-math training comparison has not run.**

Comparing several untrained models only helps select students. It cannot show that a curriculum improves a model. The main experiment must apply every selection rule to every viable student and measure changes against that same student's starting checkpoint and random-training control.

## Candidate coverage

| Student | Approximate size | Pretraining lineage | Role |
|---|---:|---|---|
| [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | 4B | Qwen | Candidate student |
| [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) | 7B | Qwen, with DeepSeek distillation | Candidate student; not an independent backbone lineage |
| [Phi-4-mini-reasoning](https://huggingface.co/microsoft/Phi-4-mini-reasoning) | 3.8B | Phi | Candidate from another model family |
| [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) | 3B | SmolLM | Candidate from another pretraining lineage |

These are four candidates spanning three declared pretraining lineages, not four statistically independent models. Teachers, architectures and pretraining data can overlap. Microsoft documents DeepSeek-generated training data for Phi as well; this is not independent training-data provenance. Model-card scores are not results from this project.

Pinned 4-bit screening releases, checksums and native decoding settings are in `comparison_config.json`. The original Qwen2.5-Math-1.5B calibration is a historical reference, not evidence of method effectiveness. Phi and SmolLM3 are configured but have not been downloaded or evaluated in this environment.

## Experiment matrix

For **each** viable student:

1. Evaluate its initial checkpoint before experiment-specific training.
2. Train separate copies with random, fixed easy-to-hard and adaptive selection.
3. Repeat every selection rule with seeds 7, 19 and 41.
4. Evaluate all conditions on the same frozen held-out questions, with evaluation settings fixed within that model.

If all four candidates pass feasibility checks, this is **4 models × 3 policies × 3 seeds = 36 proposed training runs**, plus initial-checkpoint evaluations. No such run has been completed. `study_matrix.py` generates `study_plan.json` as a planning manifest, not a training launcher. Budgets remain unset until training-only profiling and dataset preparation are complete.

At least three viable students spanning two pretraining lineages are required for the planned cross-model analysis. A model that fails the small feasibility screen needs a larger validation check or a documented replacement; an uninformative floor score is not a useful training-effect estimate. Screening never uses the reserved test set. The present 9/9/13 training/validation/test pool is exploratory and must be expanded with trustworthy solutions before a convincing main study.

## Controls

- Within a model, every policy starts from the identical pinned checkpoint and fresh adapter/optimizer state. Different models have their own initial checkpoints.
- Match supervised training-token budgets across policies within each model. Preserve complete targets; report actual token counts, example counts and raw-text lengths because tokenizers differ across models.
- Use the same eligible hard training pool and held-out split. A fixed curriculum orders relatively easier to harder examples within this hard pool; it does not introduce elementary arithmetic problems.
- Keep optimizer, adapter capacity, evaluation decoding and generation budget fixed across policies within each model. Freeze any model-specific choices using training-derived validation data before final-test evaluation.
- Account for training, selection/scoring, evaluation and solution-generation cost separately. No test correctness may influence selection or tuning.
- Repair diagram handling consistently before the final evaluation protocol. Verify answer keys and solutions; disclose unresolved semantic duplicates and pretraining overlap.

## What would demonstrate an improvement?

For each model, report:

- **Training gain:** held-out accuracy after training minus accuracy of that model's initial checkpoint.
- **Adaptive advantage:** held-out accuracy after adaptive selection minus accuracy after matched-budget random selection; also compare against the fixed curriculum.
- Seed-level variability, question-level paired outcomes, completion rates, question-type denominators and total training-plus-selection cost.

An increase over the starting checkpoint alone does not establish that adaptive selection beats ordinary training. Higher raw accuracy in a different model does not demonstrate a selection-method benefit either.

Publish individual model effects before any aggregate. A gain only in one student supports a model-specific observation. Consistent benefits in the tested families would support a bounded cross-model finding, not a universal guarantee. If benefits disappear or reverse elsewhere, report that heterogeneity and any extra cost honestly. No novelty or significance claim is assumed in advance.
