"""Render measured AMC validation results without rerunning the model."""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
    with plt.rc_context({
        'font.family': 'DejaVu Sans',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.titleweight': 'bold',
        'axes.labelcolor': '#475569',
        'text.color': '#0f172a',
        'svg.fonttype': 'none',
        'svg.hashsalt': 'amc-measurements',
    }):
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        labels = [row['url'].split('/')[-1].replace('Problem_', 'P') for row in rows]
        bars = axes[0].bar(labels, [row['tokens'] for row in rows], color='#2563eb', width=.55)
        axes[0].bar_label(bars, padding=5)
        axes[0].axhline(
            calibration['max_new_tokens'], linestyle='--', color='#94a3b8',
            label=f"Configured ceiling: {calibration['max_new_tokens']:,} tokens",
        )
        completed = sum(row['ended_with_eos'] for row in rows)
        axes[0].set(ylabel='Generated tokens', title=f'{completed}/{len(rows)} answers completed')
        axes[0].legend(frameon=False, fontsize=8)
        speeds = [row['tokens_per_second'] for row in rows]
        bars = axes[1].bar(labels, speeds, color='#0d9488', width=.55)
        axes[1].bar_label(bars, fmt='%.2f', padding=5)
        axes[1].set(ylabel='Tokens per second', title='Measured CPU speed',
                    ylim=(0, max(speeds) * 1.2))
        for axis in axes:
            axis.set_axisbelow(True)
            axis.grid(axis='y', color='#e2e8f0', linewidth=.7)
        figure.suptitle('AMC12 validation · Qwen2.5-Math-1.5B',
                       fontweight='bold', fontsize=15)
        correct = sum(row['correct'] for row in rows)
        figure.text(
            .5, .025,
            f'{len(rows)} adapted hard problems · {correct} correct · No weights updated · No test evaluation',
            ha='center', fontsize=9, color='#64748b',
        )
        figure.tight_layout(rect=(0, .07, 1, .93))
        svg_path = root / 'calibration.svg'
        figure.savefig(svg_path, metadata={'Date': None})
        svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines()) + '\n')
        figure.savefig(root / 'calibration.png', dpi=160)
        plt.close(figure)


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
