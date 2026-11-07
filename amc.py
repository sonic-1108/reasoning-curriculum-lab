"""Acquire AMC-derived problems locally and calibrate a pinned math student."""
import argparse
import hashlib
import json
import re
import time
import urllib.request
from collections import Counter
from decimal import Decimal
from pathlib import Path

import pyarrow.parquet as parquet
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_REVISION = '69d78a4a2c840e82d69af6bc742bda09005f6316'
MODEL = 'Qwen/Qwen2.5-Math-1.5B-Instruct'
MODEL_REVISION = 'aafeb0fc6f22cbf0eaeed126eff8be45b0360a35'


def source_identity(url):
    match = re.search(r'/(202[23])_AMC_12([AB])_Problems/Problem_(\d+)$',url)
    if not match:
        raise ValueError(f'Unrecognized source identity: {url}')
    year, form, number = match.groups()
    return int(year), form, int(number)


def final_number(text):
    matches = re.findall(r'\\boxed\{\s*([-+]?\d+(?:\.\d+)?)\s*\}',text)
    if not matches:
        matches = re.findall(r'(?m)^\s*####\s*([-+]?\d+(?:\.\d+)?)\s*$',text)
    return Decimal(matches[-1]) if matches else None


def acquire():
    local = Path('work/amc')
    public = Path('amc_results')
    local.mkdir(parents=True,exist_ok=True)
    public.mkdir(exist_ok=True)
    path = local/'source.parquet'
    if not path.exists():
        urllib.request.urlretrieve(f'https://huggingface.co/datasets/AI-MO/aimo-validation-amc/resolve/{DATA_REVISION}/data/train-00000-of-00001.parquet',path)
    rows = parquet.read_table(path).to_pylist()
    assert len(rows)==83
    seen = set()
    records=[]
    for row in rows:
        year,form,number=source_identity(row['url'])
        digest=hashlib.sha256(re.sub(r'\s+',' ',row['problem']).lower().strip().encode()).hexdigest()
        if digest in seen:
            raise ValueError('Duplicate normalized question across source rows')
        seen.add(digest)
        split = 'train' if (year,form)==(2022,'A') else ('validation' if year==2022 else 'reserved_test')
        records.append(dict(**row,year=year,form=form,number=number,digest=digest,split=split,eligible=number>=16))
    local.joinpath('data.json').write_text(json.dumps(records,indent=2))
    manifest=dict(dataset='AI-MO/aimo-validation-amc',revision=DATA_REVISION,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_rows=len(rows),
        variant='AMC12-derived modified integer-answer problems; not original multiple choice',
        labels='Publisher-provided adapted numeric answers; no worked solutions; not independently reverified',
        declared_license='apache-2.0 on dataset card; underlying AMC rights not independently established; no problem text redistributed',
        all_counts=dict(Counter(f"{row['year']}_AMC12{row['form']}" for row in records)),
        eligible_counts=dict(Counter(row['split'] for row in records if row['eligible'])),
        threshold='Original problem numbers 16–25 only; position is a difficulty proxy, not a guarantee',
        deduplication='83 distinct whitespace-normalized question SHA256 hashes; semantic duplicates and pretraining overlap unknown',
        non_test_records=[{key:row[key] for key in ['url','year','form','number','digest','split','eligible']} for row in records if row['split']!='reserved_test'])
    public.joinpath('manifest.json').write_text(json.dumps(manifest,indent=2))
    return records


def calibrate(rows, count, seconds):
    validation=sorted([row for row in rows if row['split']=='validation' and row['eligible']],key=lambda row:row['digest'])[:count]
    assert len(validation)==count
    torch.set_num_threads(4)
    started=time.perf_counter()
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_REVISION)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=MODEL_REVISION,torch_dtype=torch.float32)
    model.eval()
    loading_seconds=time.perf_counter()-started
    results=[]
    for row in validation:
        messages=[{'role':'system','content':'You are a helpful assistant.'},
            {'role':'user','content':row['problem']+'\nPlease reason step by step, and put your final answer within \\boxed{}.'}]
        text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=tokenizer(text,return_tensors='pt')
        started=time.perf_counter()
        with torch.no_grad():
            output=model.generate(**inputs,max_new_tokens=2048,max_time=seconds,do_sample=False,pad_token_id=tokenizer.eos_token_id)
        elapsed=time.perf_counter()-started
        continuation=output[0,inputs['input_ids'].shape[1]:]
        generated=tokenizer.decode(continuation,skip_special_tokens=True)
        predicted=final_number(generated)
        terminated=bool(len(continuation) and int(continuation[-1]) in model.generation_config.eos_token_id) if isinstance(model.generation_config.eos_token_id,list) else bool(len(continuation) and int(continuation[-1])==model.generation_config.eos_token_id)
        result=dict(url=row['url'],digest=row['digest'],split='validation',prediction=str(predicted),
            reference=str(row['answer']),correct=predicted is not None and predicted==Decimal(str(row['answer'])),
            parsed=predicted is not None,tokens=len(continuation),elapsed_seconds=elapsed,
            tokens_per_second=len(continuation)/elapsed,ended_with_eos=terminated,time_limit_reached=elapsed>=seconds)
        results.append(result)
        Path(f"work/amc/generation_{row['year']}_{row['form']}_{row['number']}.txt").write_text(generated)
        Path('amc_results/calibration.json').write_text(json.dumps(dict(model=MODEL,revision=MODEL_REVISION,
            device='cpu',dtype='float32',threads=4,loading_seconds=loading_seconds,
            max_new_tokens=2048,max_seconds_per_problem=seconds,planned_count=count,
            complete=len(results)==count,results=results),indent=2))
        print(json.dumps(result),flush=True)
    passed=sum(row['parsed'] and row['ended_with_eos'] for row in results)>=2 and sum(row['correct'] for row in results)>=1 and not any(row['time_limit_reached'] for row in results)
    Path('amc_results/GATE.json').write_text(json.dumps(dict(passed=passed,
        criterion='At least two parseable EOS completions and one correct out of three validation problems, with no time limit reached. A feasibility screen, not a benchmark claim.',
        test_ids_frozen=False,test_evaluated=False,training_performed=False),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['prepare','calibrate'],default='prepare')
    parser.add_argument('--count',type=int,choices=[3],default=3)
    parser.add_argument('--seconds',type=int,default=90)
    args=parser.parse_args()
    records=acquire()
    if args.phase=='calibrate':
        calibrate(records,args.count,args.seconds)
