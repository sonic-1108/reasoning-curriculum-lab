# What should a reasoning model learn next?

[![Data invariants](https://github.com/sonic-1108/reasoning-curriculum-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/sonic-1108/reasoning-curriculum-lab/actions/workflows/tests.yml)

## Project goal

**Find out which training problems help open-weight models improve most on hard, unseen mathematical questions for a fixed training budget.**

The intended outcome is a reproducible comparison of three ways to choose the next training example:

| Selection rule | What the student studies next |
|---|---|
| Random | A randomly selected training problem |
| Fixed curriculum | Problems ordered from relatively easier to harder within the hard-problem pool |
| Adaptive | Problems selected using the student's current errors or estimated learnability |

Within each student model, start all policies from that model's identical pretrained checkpoint and match supervised training-token budgets. Compare each trained student against its own initial checkpoint and its matched-budget random-training control on held-out questions. Report per-model gains, variation across seeds, and training and selection costs separately.

**Cross-model scope (planned):** Qwen3-4B-Instruct, DeepSeek-R1-Distill-Qwen-7B, Phi-4-mini-reasoning (3.8B), and SmolLM3-3B. These cover three declared pretraining lineages; DeepSeek's Qwen variant is not an independent backbone family. If all four qualify, **4 students × 3 policies × 3 seeds = 36 proposed training runs**, plus initial-checkpoint evaluations. None of these main-study runs has been completed. [Cross-model controls and evidence requirements](STUDY_DESIGN.md) · [Proposed matrix](study_plan.json).

**Claim boundary:** comparing untrained models only selects students. Method effectiveness requires a within-model advantage over random training, replicated across multiple viable students. A gain in one student supports a model-specific finding, not a general improvement claim.

**Why this matters:** selecting better examples could make mathematical post-training more efficient. The evidence must show whether an accuracy gain justifies the extra selection cost. A well-supported finding that adaptive selection offers no worthwhile benefit is also a valid outcome.

**Final deliverable:** saved predictions, learning curves, per-question-type results, cost comparisons and reproducible commands supporting a clear conclusion about when each selection rule helps or fails. No improvement or novel method is assumed in advance.

**Current stage:** model and data feasibility checks. The three-question AMC calibration below does **not** answer the main research question; the planned hard-math training comparison is still incomplete.

## Progress so far

**Initial result:** Qwen2.5-Math-1.5B-Instruct scored **0/3** on a small hard-AMC validation sample. All three outputs completed normally. No AMC training, final-test evaluation or curriculum comparison has been run.

**Screening blocked on GPU access:** the four-candidate runner is configured. Qwen3 and DeepSeek weights are cached; Phi and SmolLM3 have pinned configurations but have not been downloaded or evaluated. There are no completed new-candidate accuracy results. [Backend limitation and prepared GPU launcher](MODEL_COMPARISON.md).

## Which question types worked best?

| Question type | What was tested | Correct / attempted | Accuracy |
|---|---|---:|---:|
| Counting / combinatorics | Binary arrays and grid-update configurations | 0/2 | 0% |
| Geometry | Area of a square-and-hexagon configuration | 0/1 | 0% |

**No best or worst category is established.** Both observed categories scored zero. Two counting examples and one geometry example cannot support a ranking; other categories remain untested. The geometry diagram was not rendered, so its result also has an input limitation. Categories were manually assigned after the run for descriptive reporting, not used to choose training data.

![Answers, question types and measured generation time](amc_results/calibration.svg)

[Per-problem results](amc_results/RESULTS.md) · [Numeric predictions](amc_results/calibration.json) · [Category annotations](amc_results/problem_types.json) · [Generated category summary](amc_results/type_summary.json)

## Data and setup

- **Source:** [AI-MO AMC validation dataset](https://huggingface.co/datasets/AI-MO/aimo-validation-amc), 83 integer-answer adaptations of 2022/2023 AMC12 problems, not original multiple-choice papers or AIME.
- **Hard subset:** original positions 16–25 only. There are 9 training candidates from 2022A, 9 validation problems from 2022B, and 13 reserved test problems from 2023.
- **Actual evaluation:** three validation problems chosen beforehand by question hash. Greedy CPU inference; at most 2,048 new tokens and 90 seconds per answer. Observed times: 57–72 seconds.
- **Labels and limitations:** publisher-provided numeric keys, no worked solutions. Unknown pretraining overlap; no overall AMC accuracy claim. Problem text stays local because underlying redistribution rights are not independently established.

The three evaluated keys have now been independently checked: finite enumeration for counting and exact area from the supplied geometry coordinates. All agree with the dataset. This does not verify the other source keys or the original diagram transcription. [Verification record](amc_results/answer_verification.json) · [Solver code](verify_amc.py).

[Exact provenance and method](AMC_STUDY.md) · [Split manifest](amc_results/manifest.json) · [Study overview](amc_results/overview.svg)

## Reproduce

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-amc.txt
python -m unittest -v
python amc_report.py       # Audit and redraw saved results; no model run
python plot_overview.py    # Redraw the study map from saved records
```

| Core code | Purpose |
|---|---|
| [amc.py](amc.py) | Acquire data locally and run the pinned math model |
| [amc_analysis.py](amc_analysis.py) | Validate annotations and compute per-type denominators |
| [amc_report.py](amc_report.py), [plot_overview.py](plot_overview.py), [figure_style.py](figure_style.py) | Reproduce reports and figures from saved evidence |

## What remains?

Validate a more capable student and obtain a larger pool of trustworthy training solutions before comparing random, easy-to-hard and adaptive selection. Keep initial weights and supervised-token budgets matched; account for scoring costs separately. The main research question remains unanswered.

## Historical evidence

[Earlier GSM8K and arithmetic pilots](HISTORICAL_RESULTS.md) are archived separately with their methods and raw measurements. They used a different, smaller student and do not establish AMC improvements. One duplicate arithmetic dataset was replaced with an explicit checksum-verified reference; unique experimental records remain intact.

Code uses the [MIT license](LICENSE). Dataset and model rights are described in the [study method](AMC_STUDY.md).
