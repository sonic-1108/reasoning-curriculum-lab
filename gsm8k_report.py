"""Audit complete GSM8K runs and draw figures solely from saved measurements."""
import json
import platform
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from gsm8k import extract, numeric

root = Path('gsm8k_results')
assert (root / 'COMPLETE').exists(), 'Training is not complete'
manifest = json.loads((root / 'manifest.json').read_text())
rows = json.loads((root / 'predictions.json').read_text())
checkpoints = defaultdict(list)
for row in rows:
    checkpoints[(row['seed'],row['policy'],row['step'])].append(row)
for checkpoint in checkpoints.values():
    assert len(checkpoint) == manifest['config']['test_size']
    assert {row['index'] for row in checkpoint} == set(manifest['evaluation_indices'])
for row in rows:
    answer = extract(row['prediction'])
    row['parsed'] = str(answer)
    row['format_valid'] = answer is not None
    row['correct'] = answer is not None and answer == numeric(row['gold'])
(root / 'predictions.json').write_text(json.dumps(rows, indent=2))
selections = json.loads((root / 'selections.json').read_text())
pool_ids = {row['id'] for row in manifest['pool']}
assert all(row['id'] in pool_ids for row in selections)
policies = ['random','easy_to_hard','adaptive']
steps = manifest['config']['steps']
for seed in manifest['config']['seeds']:
    budgets = []
    for policy in policies:
        selected = [row for row in selections if row['seed']==seed and row['policy']==policy]
        assert len(selected) == steps
        budgets.append([row['target_tokens'] for row in selected])
        final = [row for row in rows if row['seed']==seed and row['policy']==policy and row['step']==steps]
        assert len(final) == manifest['config']['test_size']
        assert {row['index'] for row in final} == set(manifest['evaluation_indices'])
        assert all(row['supervised_tokens']==sum(budgets[-1]) for row in final)
    assert budgets[0]==budgets[1]==budgets[2]
summary=[]
for policy in ['base']+policies:
    selected=[row for row in rows if row['policy']==policy and (policy=='base' or row['step']==steps)]
    summary.append(dict(policy=policy,correct=sum(row['correct'] for row in selected),observations=len(selected),
        accuracy=np.mean([row['correct'] for row in selected]),
        format_valid=np.mean([row['format_valid'] for row in selected]),
        generation_cap_fraction=np.mean([row['generation_tokens']==manifest['config']['max_new_tokens'] for row in selected]),
        training_seconds=np.mean([row['training_seconds'] for row in selected]),
        selection_seconds=np.mean([row['selection_seconds'] for row in selected]),
        correct_by_seed={str(seed):sum(row['correct'] for row in selected if row['seed']==seed) for seed in sorted({row['seed'] for row in selected})},
        supervised_tokens_by_seed={str(seed):next(row['supervised_tokens'] for row in selected if row['seed']==seed) for seed in sorted({row['seed'] for row in selected})}))
(root/'summary.json').write_text(json.dumps(summary,indent=2))
table = ['# GSM8K fixed-subset measurements', '', f"{manifest['config']['test_size']} shared test problems; counts across seeds are repeated observations, not independent problems.", '',
    '| Policy | Final answer accuracy | Correct by seed | Valid final answer | Token cap reached | Train seconds | Select seconds |',
    '|---|---:|---|---:|---:|---:|---:|']
for row in summary:
    table.append(f"| {row['policy']} | {row['accuracy']:.1%} | {row['correct_by_seed']} | {row['format_valid']:.1%} | {row['generation_cap_fraction']:.1%} | {row['training_seconds']:.2f} | {row['selection_seconds']:.2f} |")
table.extend(['', 'Training and selection seconds are means across seeds; evaluation time is saved separately in costs.json.',
    'Correct final numbers do not certify correct explanations. Some matches occur with invalid reasoning or at the generation limit.',
    '', 'Supervised tokens per seed (identical across policies):', str(summary[1]['supervised_tokens_by_seed'])])
(root/'RESULTS.md').write_text('\n'.join(table)+'\n')
(root/'environment.json').write_text(json.dumps(dict(python=sys.version,platform=platform.platform(),
    torch=torch.__version__,device='cpu',threads=4,scorer='complete numeric marker line or boxed/explicit/bare answer',
    full_benchmark=False),indent=2))
figures=root/'figures'
figures.mkdir(exist_ok=True)
for budget in ['supervised_tokens','cost']:
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for policy in policies:
        for axis, metric in zip(axes,['correct','format_valid']):
            points=[]
            for step in sorted({row['step'] for row in rows if row['policy']==policy}):
                selected=[row for row in rows if row['policy']==policy and row['step']==step]
                points.append((np.mean([row[budget] if budget!='cost' else row['training_seconds']+row['selection_seconds'] for row in selected]),np.mean([row[metric] for row in selected])))
            base=next(row for row in summary if row['policy']=='base')
            points.insert(0,(0,base['accuracy' if metric=='correct' else 'format_valid']))
            axis.plot(*zip(*points),marker='o',label=policy)
            upper = max(.1, max(point[1] for point in points) * 1.2) if metric == 'correct' else 1.02
            axis.set(xlabel='Supervised training tokens' if budget!='cost' else 'Training + selection seconds',ylabel='Accuracy (zoomed scale)' if metric=='correct' else 'Parseable final-answer fraction',ylim=(-.002,upper))
    axes[0].legend()
    fig.suptitle(f"GSM8K: fixed {manifest['config']['test_size']}-question subset, {len(manifest['config']['seeds'])} seeds")
    fig.tight_layout()
    fig.savefig(figures/f'{budget}.svg')
    fig.savefig(figures/f'{budget}.png',dpi=160)
    plt.close(fig)
print(json.dumps(summary,indent=2))
print(f'PASS: {len(rows)} saved predictions; exact matched supervision schedules and complete evaluation checkpoints')
