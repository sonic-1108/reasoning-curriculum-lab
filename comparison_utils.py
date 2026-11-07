"""Validate frozen AMC inputs and grade only final numeric responses."""
import hashlib
import json
import re
from decimal import Decimal


def frozen_validation(root):
    baseline = json.loads((root / 'amc_results/calibration.json').read_text())
    local = json.loads((root / 'work/amc/data.json').read_text())
    lookup = {row['url']: row for row in local}
    selected = []
    for reference in baseline['results']:
        row = lookup[reference['url']]
        normalized = re.sub(r'\s+', ' ', row['problem']).lower().strip()
        digest = hashlib.sha256(normalized.encode()).hexdigest()
        if row['split'] != 'validation' or row['year'] != 2022 or row['form'] != 'B':
            raise ValueError('Only frozen 2022B validation inputs are permitted')
        if not row['eligible'] or digest != reference['digest']:
            raise ValueError('Frozen validation question changed')
        if Decimal(str(row['answer'])) != Decimal(reference['reference']):
            raise ValueError('Frozen answer key changed')
        selected.append(row)
    if len(selected) != 3 or len({row['url'] for row in selected}) != 3:
        raise ValueError('Expected exactly three distinct frozen validation problems')
    return selected


def extract_answer(text, thinking=False):
    if thinking:
        if '</think>' not in text:
            return None
        text = text.rsplit('</think>', 1)[-1]
    boxed = re.findall(r'\\boxed\{\s*([-+]?\d+(?:\.\d+)?)\s*\}', text)
    marked = re.findall(r'(?m)^\s*####\s*([-+]?\d+(?:\.\d+)?)\s*$', text)
    matches = boxed or marked
    return Decimal(matches[-1]) if matches else None
