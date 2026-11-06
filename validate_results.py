"""Audit a completed run without reloading the model."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else 'results')
data = json.loads((root / 'data.json').read_text())
metrics = json.loads((root / 'metrics.json').read_text())
selections = json.loads((root / 'selections.json').read_text())
environment = json.loads((root / 'environment.json').read_text())
train_ids = {row['id'] for row in data if row['split'] == 'train'}
evaluation_ids = set(environment['evaluation_ids'])
assert not train_ids & evaluation_ids
assert len(data) == len({row['id'] for row in data})
assert all(set(row['ids']) <= train_ids and len(row['ids']) == 8 for row in selections)
for seed in environment['config']['seeds']:
    initial_predictions = []
    for policy in ['random', 'easy_to_hard', 'adaptive']:
        selected = [row for row in metrics if row['seed'] == seed and row['policy'] == policy]
        for step in {row['step'] for row in selected}:
            checkpoint = [row for row in selected if row['step'] == step]
            assert len(checkpoint) == len(evaluation_ids)
            assert {row['id'] for row in checkpoint} == evaluation_ids
            assert all(row['slots'] == step * 256 for row in checkpoint)
        assert max(row['step'] for row in selected) == environment['config']['steps']
        initial_predictions.append({row['id']: row['prediction'] for row in selected if row['step'] == 0})
    assert initial_predictions[0] == initial_predictions[1] == initial_predictions[2]
print(f'PASS: {len(metrics)} predictions, disjoint splits, identical initial predictions, complete matched-budget runs, training-only selections')
