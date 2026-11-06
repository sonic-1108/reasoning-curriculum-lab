"""Regenerate figures and measurement tables from saved predictions."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(sys.argv[1] if len(sys.argv) > 1 else 'results')
rows = json.loads((root / 'metrics.json').read_text())
figures = root / 'figures'
figures.mkdir(exist_ok=True)
policies = ['random', 'easy_to_hard', 'adaptive']
final_step = max(row['step'] for row in rows)
summary = []
for axis in ['slots', 'cost']:
    fig, plot = plt.subplots(figsize=(7, 4))
    for policy in policies:
        points = []
        for step in sorted({row['step'] for row in rows}):
            selected = [row for row in rows if row['policy'] == policy and row['step'] == step]
            if not selected:
                continue
            budget = np.mean([row['slots'] if axis == 'slots' else row['training_seconds'] + row['scoring_seconds'] for row in selected])
            points.append((budget, np.mean([row['correct'] for row in selected])))
        plot.plot(*zip(*points), marker='o', label=policy)
    plot.set(xlabel='Padded training token slots' if axis == 'slots' else 'Training + selection seconds (evaluation excluded)', ylabel='Exact answer accuracy', title='SmolLM2-135M arithmetic pilot')
    plot.legend()
    fig.tight_layout()
    fig.savefig(figures / f'{axis}.svg')
    plt.close(fig)
for group in ['family', 'difficulty']:
    fig, plot = plt.subplots(figsize=(7, 4))
    groups = sorted({row[group] for row in rows})
    for index, policy in enumerate(policies):
        values = [np.mean([row['correct'] for row in rows if row['policy'] == policy and row['step'] == final_step and row[group] == value]) for value in groups]
        plot.bar(np.arange(len(groups)) + index * .25, values, width=.25, label=policy)
    plot.set(xticks=np.arange(len(groups)) + .25, xticklabels=groups, ylabel='Exact answer accuracy', title=f'Final accuracy by {group}')
    plot.legend()
    fig.tight_layout()
    fig.savefig(figures / f'{group}.svg')
    plt.close(fig)
for policy in policies:
    final = [row for row in rows if row['policy'] == policy and row['step'] == final_step]
    summary.append(dict(policy=policy, observations=len(final), accuracy=np.mean([row['correct'] for row in final]),
                        slots=final[0]['slots'], mean_answer_tokens=np.mean([row['answer_tokens'] for row in final]),
                        mean_training_seconds=np.mean([row['training_seconds'] for row in final]),
                        mean_selection_seconds=np.mean([row['scoring_seconds'] for row in final])))
(root / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
