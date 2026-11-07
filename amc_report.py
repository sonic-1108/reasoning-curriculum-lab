"""Render measured AMC validation results without rerunning the model."""
import json
from pathlib import Path

from figure_style import MUTED, PAPER, RED, RULE, TEAL, canvas, label, rectangle, rule, save


def load_results(root):
    calibration = json.loads((root / 'calibration.json').read_text())
    gate = json.loads((root / 'GATE.json').read_text())
    rows = calibration['results']
    assert calibration['complete'] and len(rows) == 3
    assert all(
        row['split'] == 'validation' and '/2022_AMC_12B_' in row['url']
        for row in rows
    )
    return calibration, gate, rows


def draw_measurements(root, calibration, rows):
    ordered = sorted(rows, key=lambda row: int(row['url'].split('_')[-1]))
    figure, axis = canvas(7.4, 'amc-validation-brief')
    label(axis, .055, .95, 'A M C 1 2   /   V A L I D A T I O N   B R I E F', 8, MUTED)
    label(axis, .055, .87, 'Complete answers. Incorrect results.', 24, weight='bold')
    label(axis, .055, .812, 'Qwen2.5-Math-1.5B · Three hard 2022B problems · Greedy CPU inference', 10, MUTED)
    correct = sum(row['correct'] for row in rows)
    completed = sum(row['ended_with_eos'] for row in rows)
    mean_seconds = sum(row['elapsed_seconds'] for row in rows) / len(rows)
    for horizontal, value, description, color in [
        (.055, f'{correct}/{len(rows)}', 'numeric answers correct', RED),
        (.365, f'{completed}/{len(rows)}', 'outputs ended normally', TEAL),
        (.675, f'{mean_seconds:.1f}s', 'mean time per problem', '#1c2b33'),
    ]:
        label(axis, horizontal, .716, value, 29, color, 'bold')
        label(axis, horizontal, .657, description, 9, MUTED)
    rule(axis, (.055, .61), (.945, .61))
    label(axis, .055, .565, '01   FINAL-ANSWER AUDIT', 9, weight='bold')
    label(axis, .595, .565, '02   TIME TO ANSWER', 9, weight='bold')
    for horizontal, text in [(.07, 'PROBLEM'), (.255, 'MODEL'), (.375, 'REFERENCE')]:
        label(axis, horizontal, .507, text, 7.5, MUTED)
    limit = calibration['max_seconds_per_problem']
    for index, row in enumerate(ordered):
        vertical = .435 - index * .105
        if index % 2 == 0:
            rectangle(axis, (.055, vertical - .037), .455, .075, PAPER)
        number = row['url'].split('_')[-1]
        reference = f"{float(row['reference']):g}"
        label(axis, .07, vertical, f'Problem {number}', 10)
        label(axis, .255, vertical, row['prediction'], 17, RED, 'bold')
        label(axis, .375, vertical, reference, 17, weight='bold')
        start = .615
        endpoint = start + .26 * row['elapsed_seconds'] / limit
        rule(axis, (start, vertical), (.875, vertical), RULE, 2)
        rule(axis, (start, vertical), (endpoint, vertical), TEAL, 2)
        axis.scatter([endpoint], [vertical], s=42, color=TEAL, zorder=3)
        label(axis, .94, vertical, f"{row['elapsed_seconds']:.1f}s", 10, weight='bold', ha='right')
        label(axis, .615, vertical - .035,
              f"{row['tokens']} tokens  ·  {row['tokens_per_second']:.2f} tokens/s", 8, MUTED)
    rule(axis, (.875, .18), (.875, .485), color=RULE, linestyle='--')
    for fraction in [0, .5, 1]:
        label(axis, .615 + .26 * fraction, .154, f'{limit * fraction:g}s', 8, MUTED, ha='center')
    label(axis, .055, .111, 'READOUT', 8, RED, 'bold')
    label(axis, .145, .111, 'This configuration failed the capability screen; three trials do not estimate overall AMC accuracy.', 9)
    label(axis, .055, .054, 'Validation only · No fine-tuning or test run · Adapted numeric keys · Geometry diagram was not rendered.', 8, MUTED)
    save(figure, root, 'calibration')


def write_table(root, gate, rows):
    table = [
        '# AMC12 local calibration measurements', '',
        '**Validation only. No weights updated; no final test or three-policy study run.**', '',
        '| 2022 AMC12B problem | Generated tokens | Seconds | Tokens/s | EOS completed | Correct numeric answer |',
        '|---|---:|---:|---:|---|---|',
    ]
    for row in rows:
        table.append(
            f"| [{row['url'].split('/')[-1]}]({row['url']}) | {row['tokens']} "
            f"| {row['elapsed_seconds']:.2f} | {row['tokens_per_second']:.2f} "
            f"| {row['ended_with_eos']} | {row['correct']} |"
        )
    table.extend([
        '', f"Feasibility gate passed: **{gate['passed']}**.",
        'Correctness uses publisher-provided adapted integer keys; it is not an independent proof of solution validity.',
        'Full generations stay local under ignored work/amc. Numeric predictions, identifiers and timings are published in calibration.json.',
        'A hypothetical 2,048-token output at measured speed is an extrapolation, not an observed full completion.',
        'The geometry problem includes diagram source code; no rendered image was supplied to the text-only model.',
    ])
    (root / 'RESULTS.md').write_text('\n'.join(table) + '\n')


def main():
    root = Path('amc_results')
    calibration, gate, rows = load_results(root)
    draw_measurements(root, calibration, rows)
    write_table(root, gate, rows)


if __name__ == '__main__':
    main()
