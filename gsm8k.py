"""Small, frozen-subset GSM8K curriculum study with matched supervised tokens."""
import argparse
import hashlib
import json
import random
import re
import time
import urllib.request
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_REVISION = '3101c7d5072418e28b9008a6636bde82a006892c'
MODEL_REVISION = '93efa2f097d58c2a74874c7e644dbc9b0cee75a2'


def numeric(text):
    try:
        return Decimal(text.replace(',', '').replace('$', '').strip().rstrip('.'))
    except InvalidOperation:
        return None


def extract(text):
    marked = re.findall(r'(?<!#)####(?!#)\s*([-+]?\$?[\d,]+(?:\.\d+)?)', text)
    boxed = re.findall(r'\\boxed\{([-+]?\$?[\d,]+(?:\.\d+)?)\}', text)
    if marked or boxed:
        return numeric((marked or boxed)[-1])
    explicit = re.findall(r'(?:final answer(?: is|:)?|answer is|answer:)\s*\$?([-+]?[\d,]+(?:\.\d+)?)', text, flags=re.I)
    if explicit:
        return numeric(explicit[-1])
    if re.fullmatch(r'\s*\$?[-+]?[\d,]+(?:\.\d+)?\.?\s*', text):
        return numeric(text)
    return None


def prompt(question):
    return f'Question: {question}\nSolve step by step. End with #### followed by the final number.\nAnswer:'


def training_tensors(row, pad_token_id):
    tokens = row['prefix'] + row['target']
    if len(tokens) > 256 or not row['prefix'] or not row['target']:
        raise ValueError('Complete prompt and solution must fit; no label truncation allowed')
    inputs = torch.tensor([tokens + [pad_token_id] * (256-len(tokens))])
    mask = torch.tensor([[1]*len(tokens)+[0]*(256-len(tokens))])
    labels = inputs.clone()
    labels[:, :len(row['prefix'])] = -100
    labels[:, len(tokens):] = -100
    assert int((labels[:, 1:] != -100).sum()) == len(row['target'])
    return dict(input_ids=inputs, attention_mask=mask, labels=labels)


def download(split, cache):
    path = cache / f'{split}.jsonl'
    if not path.exists():
        url = f'https://raw.githubusercontent.com/openai/grade-school-math/{DATA_REVISION}/grade_school_math/data/{split}.jsonl'
        urllib.request.urlretrieve(url, path)
    return [json.loads(line) for line in path.read_text().splitlines()], hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=12)
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 19, 41])
    parser.add_argument('--test-size', type=int, default=24)
    parser.add_argument('--pool-size', type=int, default=512)
    parser.add_argument('--max-new-tokens', type=int, default=96)
    parser.add_argument('--output', default='gsm8k_results')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(exist_ok=True)
    cache = Path('work/gsm8k')
    cache.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train, train_hash = download('train', cache)
    test, test_hash = download('test', cache)
    assert len(train) == 7473 and len(test) == 1319
    normalize = lambda text: re.sub(r'\s+', ' ', text.lower()).strip()
    test_questions = {normalize(row['question']) for row in test}
    train = [row for row in train if normalize(row['question']) not in test_questions]
    tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-135M', revision=MODEL_REVISION)
    tokenizer.pad_token = tokenizer.eos_token
    groups = defaultdict(list)
    seen = set()
    random.Random(2026).shuffle(train)
    for row in train:
        key = hashlib.sha256(normalize(row['question']).encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        prefix = tokenizer.encode(prompt(row['question']))
        target = tokenizer.encode(' ' + row['answer'], add_special_tokens=False) + [tokenizer.eos_token_id]
        if len(prefix) + len(target) > 256:
            continue
        prepared = dict(id=key, question=row['question'], answer=row['answer'], prefix=prefix, target=target,
                        difficulty=row['answer'].count('<<'))
        groups[len(target)].append(prepared)
        if sum(map(len, groups.values())) >= args.pool_size:
            break
    groups = {length: group for length, group in groups.items() if len(group) >= 4}
    evaluation = random.Random(2027).sample(list(enumerate(test)), args.test_size)
    pool = [row for group in groups.values() for row in group]
    output.joinpath('manifest.json').write_text(json.dumps(dict(config=vars(args), dataset_revision=DATA_REVISION,
        model_revision=MODEL_REVISION, train_sha256=train_hash, test_sha256=test_hash,
        source_sizes={'train':7473,'test':1319}, eligible_pool_size=len(pool),
        pool=[{key:value for key,value in row.items() if key not in ['prefix','target']} for row in pool],
        evaluation_indices=[index for index, _ in evaluation], setup_seconds=time.perf_counter()-started), indent=2))
    torch.set_num_threads(4)
    model = AutoModelForCausalLM.from_pretrained('HuggingFaceTB/SmolLM2-135M', revision=MODEL_REVISION)
    model.to('cpu')
    initial = {key:value.clone() for key,value in model.state_dict().items()}
    def loss(row):
        return model(**training_tensors(row, tokenizer.eos_token_id)).loss
    predictions, selections, costs = [], [], []
    def evaluate(seed, policy, step, supervised, training, scoring):
        model.eval()
        started = time.perf_counter()
        for index, row in evaluation:
            inputs = tokenizer(prompt(row['question']), return_tensors='pt')
            with torch.no_grad():
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                           pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(generated[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            gold = extract(row['answer'])
            answer = extract(text)
            predictions.append(dict(seed=seed, policy=policy, step=step, index=index,
                question=row['question'], gold=str(gold), prediction=text, parsed=str(answer),
                correct=answer is not None and answer == gold, format_valid=answer is not None,
                supervised_tokens=supervised, padded_slots=step*256,
                training_seconds=training, selection_seconds=scoring,
                generation_tokens=generated.shape[1]-inputs['input_ids'].shape[1]))
        costs.append(dict(seed=seed,policy=policy,step=step,evaluation_seconds=time.perf_counter()-started))
        output.joinpath('predictions.json').write_text(json.dumps(predictions, indent=2))
        print(seed, policy, step, 'correct',sum(row['correct'] for row in predictions[-args.test_size:]),flush=True)
    evaluate(0, 'base', 0, 0, 0, 0)
    for seed in args.seeds:
        schedule = random.Random(seed).choices(sorted(groups), k=args.steps)
        for policy in ['random', 'easy_to_hard', 'adaptive']:
            model.load_state_dict(initial)
            torch.manual_seed(seed)
            rng = random.Random(seed)
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
            supervised = 0
            training = scoring = 0.0
            for step, length in enumerate(schedule):
                started = time.perf_counter()
                candidates = rng.sample(groups[length], min(4,len(groups[length])))
                if policy == 'easy_to_hard':
                    candidates.sort(key=lambda row: abs(row['difficulty']-(1+5*step/max(1,args.steps-1))))
                elif policy == 'adaptive':
                    model.eval()
                    with torch.no_grad():
                        scores = [float(loss(row)) for row in candidates]
                    candidates = [row for _,row in sorted(zip(scores,candidates),key=lambda pair:abs(pair[0]-1.5))]
                row = candidates[0]
                scoring += time.perf_counter()-started
                started = time.perf_counter()
                model.train()
                optimizer.zero_grad()
                loss(row).backward()
                optimizer.step()
                training += time.perf_counter()-started
                supervised += length
                selections.append(dict(seed=seed,policy=policy,step=step,id=row['id'],target_tokens=length))
                if (step+1) % 6 == 0 or step+1 == args.steps:
                    evaluate(seed,policy,step+1,supervised,training,scoring)
            output.joinpath('selections.json').write_text(json.dumps(selections,indent=2))
    output.joinpath('costs.json').write_text(json.dumps(costs,indent=2))
    output.joinpath('COMPLETE').write_text('All configured runs completed.\n')


if __name__ == '__main__':
    main()
