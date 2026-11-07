# Local student comparison

Status: blocked on local GPU access. The Qwen3 CPU attempt has not yielded a scored trial; DeepSeek generation has not started. No accuracy result for either new candidate is claimed.

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

Use a separate Python 3.12 environment for the Apple-Silicon backend. The existing main environment remains unchanged. Install `requirements-mlx.txt`, prepare local inputs with the existing `amc.py --phase prepare`, then run `compare_students.py` once per candidate.

The installed Metal-enabled MLX wheel could not initialize because this execution session cannot access a GPU. A CPU-only build from the official MLX v0.32.4 source commit `36969c094354086e15d2a327a9c50feb20b9cf11` succeeded using `CMAKE_ARGS='-DMLX_BUILD_METAL=OFF -DMLX_BUILD_CPU=ON'`. The CPU workaround is documented by an [MLX maintainer](https://github.com/ml-explore/mlx/issues/3148).

The first CPU attempt loaded the Qwen3 weights, but had no scored trial after more than 11 minutes. A token-level time check cannot interrupt a long native prompt-processing operation. This is a runtime limitation, not a measured wrong answer or a new 0/3 score. `backend_probe.json` records the observed checkpoint; the slow process may still be active until the launcher stops it. The updated runner checks deadlines after smaller prompt-processing chunks as well as during generation; native operations can still overshoot a deadline until a chunk completes.

### Prepared GPU retry

Both model downloads are cached locally. A standard Metal-enabled environment is prepared separately at `work/metal-env/`; the CPU environment is retained for provenance. From a normal macOS Terminal with GPU access, run:

```bash
cd /path/to/reasoning-curriculum-lab
bash run_student_comparison.sh
```

The launcher checks GPU access, stops only the identified slow CPU attempt, preserves its incomplete record, runs both candidates sequentially, and generates a comparison from completed records. It does not purchase cloud compute. Review the saved outputs before publishing results. Recreating the GPU environment on another machine:

```bash
python3.12 -m venv work/metal-env
work/metal-env/bin/pip install -r requirements-mlx.txt
export HF_HOME="$PWD/work/hf"
work/metal-env/bin/python -c 'import json; from huggingface_hub import snapshot_download; config=json.load(open("comparison_config.json")); [snapshot_download(entry["model"], revision=entry["revision"], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"]) for entry in config["candidates"].values()]'
```

The local question cache must also exist; reproduce it with `amc.py --phase prepare` in the main environment. Model downloads are not committed to GitHub.

The planned comparison settings are unchanged; moving to the GPU changes hardware and runtime, which are recorded in each new result. No trial from the stalled CPU attempt will be silently combined with a GPU screen.

Before interpreting model mistakes, `verify_amc.py` independently checked all three numeric keys from the supplied constraints/coordinates. All match. This verification uses no reserved-test data and does not provide training solutions or update weights.
