"""Run one pinned local candidate on frozen validation questions; never train."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from decimal import Decimal
from pathlib import Path

from comparison_utils import BudgetExceeded, check_budget, extract_answer, frozen_validation


def run(candidate, weights):
    root = Path(__file__).resolve().parent
    config = json.loads((root / 'comparison_config.json').read_text())
    settings = config['candidates'][candidate]
    with (weights / 'model.safetensors').open('rb') as weight_file:
        checksum = hashlib.file_digest(weight_file, 'sha256').hexdigest()
    if checksum != settings['weights_sha256']:
        raise ValueError('Model weight checksum does not match the pinned release')
    rows = frozen_validation(root)
    output = root / 'amc_results/model_comparison'
    raw = root / 'work/model_comparison' / candidate
    output.mkdir(exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    destination = output / f'{candidate}.json'
    result = dict(
        candidate=candidate, model=settings['model'], revision=settings['revision'],
        source_model=settings['source_model'], settings=settings,
        max_new_tokens=config['max_new_tokens'],
        max_seconds_per_problem=config['max_seconds_per_problem'],
        seed=config['seed'], status='loading', complete=False,
        python=platform.python_version(), platform=platform.platform(),
        training_performed=False, test_evaluated=False, results=[],
    )

    def persist():
        destination.write_text(json.dumps(result, indent=2) + '\n')

    persist()
    try:
        import mlx.core as mx
        mx.set_default_device(mx.gpu if mx.metal.is_available() else mx.cpu)
        from mlx_lm import load, stream_generate
        from mlx_lm.sample_utils import make_sampler
    except ImportError:
        result.update(status='blocked', reason='Local MLX backend could not initialize')
        persist()
        raise
    result['device'] = str(mx.default_device())
    result['software'] = {package: importlib.metadata.version(package)
                          for package in ['mlx', 'mlx-lm', 'transformers']}
    started = time.perf_counter()
    model, tokenizer, model_config = load(str(weights), return_config=True, trust_remote_code=False)
    result['load_seconds'] = time.perf_counter() - started
    result['quantization'] = model_config.get('quantization', model_config.get('quantization_config'))
    result['status'] = 'running'
    persist()
    for index, row in enumerate(rows):
        mx.random.seed(config['seed'] + index)
        instruction = row['problem'] + '\nPlease reason step by step, and put your final answer within \\boxed{}.'
        prompt = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': instruction}], tokenize=False, add_generation_prompt=True,
        )
        if settings['thinking_prefix'] and not prompt.rstrip().endswith('<think>'):
            prompt += '<think>\n'
        sampler = make_sampler(temp=settings['temperature'], top_p=settings['top_p'], top_k=settings['top_k'])
        chunks = []
        response = None
        timeout = False
        started = time.perf_counter()
        progress_path = raw / 'progress.json'

        def prompt_progress(processed, total):
            progress_path.write_text(json.dumps(dict(
                problem=row['number'], stage='prefill', processed=processed,
                total=total, elapsed_seconds=time.perf_counter() - started,
            )))
            check_budget(started, config['max_seconds_per_problem'], time.perf_counter())

        stream = stream_generate(model, tokenizer, prompt, max_tokens=config['max_new_tokens'],
                                 sampler=sampler, prefill_step_size=32,
                                 prompt_progress_callback=prompt_progress)
        try:
            for response in stream:
                chunks.append(response.text)
                if response.finish_reason is not None:
                    break
                check_budget(started, config['max_seconds_per_problem'], time.perf_counter())
        except BudgetExceeded:
            timeout = True
        finally:
            stream.close()
        elapsed = time.perf_counter() - started
        text = ''.join(chunks)
        (raw / f"problem_{row['number']}.txt").write_text(text)
        answer = extract_answer(text, thinking=settings['thinking_prefix'])
        finished = response is not None and response.finish_reason == 'stop'
        tokens = response.generation_tokens if response is not None else 0
        trial = dict(
            url=row['url'], digest=row['digest'], split='validation',
            prediction=str(answer), reference=str(row['answer']),
            correct=answer is not None and answer == Decimal(str(row['answer'])),
            parsed=answer is not None, tokens=tokens, elapsed_seconds=elapsed,
            tokens_per_second=tokens / elapsed, ended_with_eos=finished,
            time_limit_reached=timeout,
            token_limit_reached=response is not None and response.finish_reason == 'length',
            reasoning_closed='</think>' in text if settings['thinking_prefix'] else None,
            prompt_tokens=response.prompt_tokens if response is not None else None,
            peak_memory_gb=response.peak_memory if response is not None and response.peak_memory > 0 else None,
        )
        result['results'].append(trial)
        persist()
        print(json.dumps(trial), flush=True)
    result.update(status='completed', complete=True)
    persist()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', choices=['qwen3', 'deepseek'], required=True)
    parser.add_argument('--weights', type=Path, required=True)
    arguments = parser.parse_args()
    try:
        run(arguments.candidate, arguments.weights)
    except Exception as error:
        destination = Path(__file__).resolve().parent / 'amc_results/model_comparison' / f'{arguments.candidate}.json'
        if destination.exists():
            failed = json.loads(destination.read_text())
            if failed['status'] != 'blocked':
                failed.update(status='failed', complete=False, error_type=type(error).__name__)
                destination.write_text(json.dumps(failed, indent=2) + '\n')
        raise
