# Local student comparison

Status: preparation in progress. No results for the new candidates are claimed yet.

The user approved comparing two alternative students before starting AMC training. The historical Qwen2.5-Math-1.5B run remains unchanged.

## Fixed before running

- Use exactly the three validation questions previously selected by hash: 2022 AMC12B problems 25, 17 and 18. Keep all 2023 reserved test questions unevaluated.
- Compare [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) and [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) through pinned 4-bit community MLX releases. Exact repositories and revisions are in `comparison_config.json`.
- Use local compute only. Load one model per process. Record the actual backend device, software versions, load time, quantization metadata, generation time, output tokens and completion status.
- Both candidates receive the same problem text and user instruction, without a system message. Use each publisher's decoding guidance: temperature 0.7/top-p 0.8/top-k 20 for Qwen3; temperature 0.6 for DeepSeek. DeepSeek top-p 0.95 is a declared local setting, not a publisher benchmark claim. Both use the same fixed seed schedule.
- Allow up to 16,384 new tokens and 300 seconds per problem. These are bounded local screening budgets. A timeout or token cap means incomplete generation, not proof of poor mathematical ability. DeepSeek benchmark-quality performance may require longer outputs and repeated sampling than this screen allows.
- DeepSeek uses the publisher-recommended thinking prefix. Grade only text after a closing thinking tag; an unfinished reasoning trace is not a final answer.
- Preserve raw generations locally, under ignored `work/model_comparison/`. Publish only problem identifiers, numeric predictions, correctness and timing, consistent with the source's unresolved underlying AMC redistribution rights.

## Interpretation

This compares practical local configurations, with different sampling settings and quantized weights. It does not isolate architecture, size, decoding or quantization effects. The older float32 CPU baseline also used a different prompt and shorter limits, so its runtime is not a hardware-controlled comparison.

All inputs are the same adapted integer-answer variants; the geometry diagram remains unrendered for every text-only candidate. Three problems and one sample per problem cannot establish a reliable winner. No AMC fine-tuning is performed. A useful candidate would justify a larger validation study, not an immediate claim of improved reasoning.

## Reproduce

Use a separate Python 3.12 environment for the Apple-Silicon backend. The existing main environment remains unchanged. Install `requirements-mlx.txt`, prepare local inputs with the existing `amc.py --phase prepare`, then run `compare_students.py` once per candidate. Exact commands and actual results will be added after backend validation.
