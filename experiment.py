"""Controlled answer-only arithmetic fine-tuning; all metrics come from saved runs."""
import argparse
import hashlib
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def dataset():
    rows = []
    for family in ['add', 'subtract', 'multiply', 'compose']:
        for left in range(2, 22):
            for right in range(2, 12):
                expression, answer = {
                    'add': (f'{left}+{right}', left + right),
                    'subtract': (f'{left}-{right}', left - right),
                    'multiply': (f'{left}*{right}', left * right),
                    'compose': (f'({left}+{right})*2', (left + right) * 2),
                }[family]
                prompt = f'Calculate {expression}. Answer:'
                key = hashlib.sha256(prompt.encode()).hexdigest()
                split = 'ood' if family == 'compose' else ('test' if int(key[:8], 16) % 5 == 0 else 'train')
                rows.append(dict(prompt=prompt, answer=str(answer), family=family,
                                 difficulty=1 + int(left >= 12) + int(family == 'multiply'), split=split, id=key))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='HuggingFaceTB/SmolLM2-135M')
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 19, 41])
    parser.add_argument('--output', default='results')
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = Path(args.output)
    output.mkdir(exist_ok=True)
    start = time.perf_counter()
    rows = dataset()
    output.joinpath('data.json').write_text(json.dumps(rows, indent=2))
    generation_seconds = time.perf_counter() - start
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    revision = getattr(model.config, '_commit_hash', None)
    initial = {key: value.clone() for key, value in model.state_dict().items()}
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model.to(device)
    train = [row for row in rows if row['split'] == 'train']
    evaluation = [row for row in rows if row['split'] != 'train']
    random.Random(123).shuffle(evaluation)
    evaluation = evaluation[:64]
    def encode(batch):
        texts = [row['prompt'] + ' ' + row['answer'] + tokenizer.eos_token for row in batch]
        inputs = tokenizer(texts, padding='max_length', truncation=True, max_length=32, return_tensors='pt').to(device)
        labels = inputs['input_ids'].clone()
        labels[inputs['attention_mask'] == 0] = -100
        for index, row in enumerate(batch):
            length = len(tokenizer(row['prompt'], add_special_tokens=True)['input_ids'])
            labels[index, :length] = -100
        return inputs, labels
    def losses(batch):
        inputs, labels = encode(batch)
        logits = model(**inputs).logits[:, :-1]
        targets = labels[:, 1:]
        values = torch.nn.functional.cross_entropy(logits.transpose(1, 2), targets, reduction='none', ignore_index=-100)
        return values.sum(1) / (targets != -100).sum(1)
    records = []
    selections = []
    for seed in args.seeds:
        for policy in ['random', 'easy_to_hard', 'adaptive']:
            model.load_state_dict(initial)
            torch.manual_seed(seed)
            rng = random.Random(seed)
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
            training_seconds = scoring_seconds = evaluation_seconds = 0.0
            answer_tokens = 0
            for step in range(args.steps + 1):
                if step % 10 == 0 or step == args.steps:
                    model.eval()
                    started = time.perf_counter()
                    for row in evaluation:
                        inputs = tokenizer(row['prompt'], return_tensors='pt').to(device)
                        with torch.no_grad():
                            prediction = model.generate(**inputs, max_new_tokens=6, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                        generated = tokenizer.decode(prediction[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
                        records.append(dict(seed=seed, policy=policy, step=step, slots=step*8*32,
                                            answer_tokens=answer_tokens, family=row['family'], difficulty=row['difficulty'],
                                            split=row['split'], id=row['id'], prediction=generated,
                                            correct=generated == row['answer'], training_seconds=training_seconds,
                                            scoring_seconds=scoring_seconds))
                    evaluation_seconds += time.perf_counter() - started
                    output.joinpath('metrics.json').write_text(json.dumps(records))
                    print(seed, policy, step, flush=True)
                if step == args.steps:
                    break
                started = time.perf_counter()
                candidates = rng.sample(train, 32)
                if policy == 'easy_to_hard':
                    level = 1 + 3 * step / args.steps
                    candidates.sort(key=lambda row: abs(row['difficulty'] - level))
                elif policy == 'adaptive':
                    model.eval()
                    with torch.no_grad():
                        scores = losses(candidates).cpu().tolist()
                    candidates = [row for _, row in sorted(zip(scores, candidates), key=lambda pair: abs(pair[0] - 1.5))]
                batch = candidates[:8]
                scoring_seconds += time.perf_counter() - started
                selections.append(dict(seed=seed, policy=policy, step=step, ids=[row['id'] for row in batch]))
                started = time.perf_counter()
                model.train()
                inputs, labels = encode(batch)
                answer_tokens += int((labels[:, 1:] != -100).sum())
                optimizer.zero_grad()
                model(**inputs, labels=labels).loss.backward()
                optimizer.step()
                if device == 'mps':
                    torch.mps.synchronize()
                training_seconds += time.perf_counter() - started
            output.joinpath('selections.json').write_text(json.dumps(selections))
    output.joinpath('environment.json').write_text(json.dumps(dict(model=args.model, revision=revision, device=device,
        platform=platform.platform(), torch=torch.__version__, config=vars(args), generation_seconds=generation_seconds,
        evaluation_ids=[row['id'] for row in evaluation], evaluation_seconds_last_run=evaluation_seconds), indent=2))


if __name__ == '__main__':
    main()
