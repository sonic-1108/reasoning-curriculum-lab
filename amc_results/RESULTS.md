# AMC12 local calibration measurements

**Validation only. No weights updated; no final test or three-policy study run.**

## Performance by question type

| Question type | Correct / attempted | Accuracy |
|---|---:|---:|
| Counting / combinatorics | 0/2 | 0% |
| Geometry | 0/1 | 0% |

**Best category: not established. Worst category: not established.** Both observed categories scored zero; two counting questions and one geometry question cannot support a ranking. Other categories were not evaluated.
Geometry has an additional input limitation: no rendered diagram was provided. Shorter runtime does not imply better mathematical performance.
Categories are manual post-hoc descriptive annotations, not official AMC tags, and did not affect selection or training.

## Per-problem measurements

| 2022 AMC12B problem | Generated tokens | Seconds | Tokens/s | EOS completed | Correct numeric answer |
|---|---:|---:|---:|---|---|
| [Problem_25](https://artofproblemsolving.com/wiki/index.php/2022_AMC_12B_Problems/Problem_25) | 558 | 72.22 | 7.73 | True | False |
| [Problem_17](https://artofproblemsolving.com/wiki/index.php/2022_AMC_12B_Problems/Problem_17) | 501 | 59.02 | 8.49 | True | False |
| [Problem_18](https://artofproblemsolving.com/wiki/index.php/2022_AMC_12B_Problems/Problem_18) | 496 | 57.44 | 8.64 | True | False |

Feasibility gate passed: **False**.
Correctness uses publisher-provided adapted integer keys; it is not an independent proof of solution validity.
Full generations stay local under ignored work/amc. Numeric predictions, identifiers and timings are published in calibration.json.
A hypothetical 2,048-token output at measured speed is an extrapolation, not an observed full completion.
The geometry problem includes diagram source code; no rendered image was supplied to the text-only model.
