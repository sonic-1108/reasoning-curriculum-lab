"""Render a compact study map from saved provenance and calibration."""
import json
from pathlib import Path

from figure_style import MUTED, PAPER, RED, TEAL, canvas, label, rectangle, rule, save


def main():
    root = Path('amc_results')
    manifest = json.loads((root / 'manifest.json').read_text())
    calibration = json.loads((root / 'calibration.json').read_text())
    gate = json.loads((root / 'GATE.json').read_text())
    counts = manifest['eligible_counts']
    rows = calibration['results']
    eligible = sum(counts.values())
    correct = sum(row['correct'] for row in rows)
    assert calibration['complete'] and not gate['passed']
    assert not gate['test_evaluated'] and not gate['training_performed']
    figure, axis = canvas(6.6, 'amc-study-map')
    label(axis, .055, .945, 'R E A S O N I N G   C U R R I C U L U M   L A B', 8, MUTED)
    label(axis, .055, .852, 'First establish a capable student.', 24, weight='bold')
    label(axis, .055, .788, 'AMC12 hard-problem study  /  Current phase: validation calibration', 10, MUTED)
    rule(axis, (.055, .735), (.945, .735))
    label(axis, .055, .675, '01   DATA DESIGN', 9, weight='bold')
    label(axis, .655, .675, '02   OBSERVED RESULT', 9, weight='bold')
    rule(axis, (.605, .245), (.605, .695))
    for horizontal, value, description in [
        (.055, manifest['source_rows'], 'downloaded'),
        (.235, eligible, 'eligible'),
        (.415, len(rows), 'attempted'),
    ]:
        label(axis, horizontal, .568, str(value), 32, weight='bold')
        label(axis, horizontal, .499, description, 10, MUTED)
    start = .055
    for split, color in [('train', '#cad4d8'), ('validation', TEAL), ('reserved_test', '#526773')]:
        width = .49 * counts[split] / eligible
        rectangle(axis, (start, .393), width, .045, color)
        label(axis, start + width / 2, .415, str(counts[split]), 9,
              '#1c2b33' if split == 'train' else 'white', weight='bold', ha='center')
        start += width
    for horizontal, title, subtitle in [
        (.055, 'Training candidates', '2022A · not trained'),
        (.235, 'Validation', '2022B · 3 of 9 tried'),
        (.415, 'Reserved test', '2023 A/B · untouched'),
    ]:
        label(axis, horizontal, .346, title, 8.5, weight='bold')
        label(axis, horizontal, .307, subtitle, 8, MUTED)
    excluded = manifest['source_rows'] - eligible
    label(axis, .055, .236, f'Filter: original positions 16–25 only; {excluded} other problems excluded.', 8.5, MUTED)
    label(axis, .655, .568, f'{correct} / {len(rows)}', 32, RED, 'bold')
    label(axis, .655, .499, 'correct numeric answers', 10, MUTED)
    completed = sum(row['ended_with_eos'] for row in rows)
    label(axis, .655, .416, f'{completed} / {len(rows)} finished normally', 13, weight='bold')
    label(axis, .655, .361, 'Qwen2.5-Math-1.5B · CPU', 9.5, MUTED)
    label(axis, .655, .308, 'No weights updated. No final test run.', 8.5, MUTED)
    rectangle(axis, (.055, .09), .89, .09, PAPER)
    rectangle(axis, (.055, .09), .003, .09, TEAL)
    label(axis, .075, .135, 'NEXT GATE', 8, TEAL, 'bold')
    label(axis, .205, .135, 'Stronger validated student + trusted solutions before curriculum training.', 9.4)
    label(axis, .055, .045, 'Adapted integer-answer variants · Not an overall AMC score · Source: saved manifest, predictions and gate decision.', 8, MUTED)
    save(figure, root, 'overview')


if __name__ == '__main__':
    main()
