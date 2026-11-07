"""Render the study overview from saved AMC measurements and provenance."""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def card(axis, position, title, body, color='#2563eb', dashed=False):
    horizontal, vertical, width, height = position
    axis.add_patch(FancyBboxPatch(
        (horizontal, vertical), width, height,
        boxstyle='round,pad=0.018,rounding_size=0.025',
        facecolor='white', edgecolor=color, linewidth=1.5,
        linestyle='--' if dashed else '-',
    ))
    axis.text(horizontal + .02, vertical + height - .045, title,
              fontsize=10.5, fontweight='bold', color=color, va='top')
    axis.text(horizontal + .02, vertical + height - .115, body,
              fontsize=9, color='#475569', va='top', linespacing=1.6)


def arrow(axis, start, end, dashed=False):
    axis.add_patch(FancyArrowPatch(
        start, end, arrowstyle='-|>', mutation_scale=14,
        linewidth=1.3, color='#94a3b8',
        linestyle='--' if dashed else '-',
    ))


def main():
    plt.rcParams.update({'svg.fonttype': 'none', 'svg.hashsalt': 'amc-overview'})
    root = Path('amc_results')
    manifest = json.loads((root / 'manifest.json').read_text())
    calibration = json.loads((root / 'calibration.json').read_text())
    gate = json.loads((root / 'GATE.json').read_text())
    counts = manifest['eligible_counts']
    rows = calibration['results']
    correct = sum(row['correct'] for row in rows)
    assert calibration['complete'] and not gate['passed']
    assert not gate['test_evaluated'] and not gate['training_performed']
    figure, axis = plt.subplots(figsize=(13, 6.5), facecolor='#f8fafc')
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis('off')
    axis.text(.025, .96, 'What should a reasoning model learn next?',
              fontsize=21, weight='bold', color='#0f172a', va='top')
    axis.text(.025, .885, 'CURRENT EVIDENCE  /  AMC12 hard-problem calibration',
              fontsize=10, color='#64748b', va='top')
    card(axis, (.04, .52, .25, .26), '01 · Source',
         f"{manifest['source_rows']} adapted AMC12 problems\n2022–2023 · Positions 16–25 only\nNumeric keys; no worked solutions")
    card(axis, (.37, .52, .25, .26), '02 · Separate by form',
         f"2022A · {counts['train']} training candidates\n2022B · {counts['validation']} validation problems\n2023 A/B · {counts['reserved_test']} reserved test problems")
    card(axis, (.70, .52, .25, .26), '03 · Calibrate the student',
         f"Qwen2.5-Math-1.5B\n{len(rows)} validation problems · {correct} correct\nAll {sum(row['ended_with_eos'] for row in rows)} outputs completed", color='#0d9488')
    arrow(axis, (.31, .65), (.35, .65))
    arrow(axis, (.64, .65), (.68, .65))
    card(axis, (.04, .12, .43, .25), 'MEASURED · Capability gate not passed',
         f"CPU inference runs; this configuration scored {correct}/{len(rows)}.\nNo claim of overall AMC accuracy or reasoning gains.", color='#b45309')
    card(axis, (.53, .12, .42, .25), 'PROPOSED · Curriculum comparison',
         'Stronger validated student + trusted training solutions\nRandom / easy-to-hard / adaptive · Matched token budgets\nNot run · Reserved test remains untouched', color='#64748b', dashed=True)
    arrow(axis, (.70, .49), (.46, .40))
    arrow(axis, (.83, .49), (.83, .40), dashed=True)
    axis.text(.025, .025, 'Source: saved manifest, calibration predictions and gate decision. Historical pilots remain available below.',
              fontsize=8.5, color='#64748b')
    figure.subplots_adjust(left=.02, right=.98, top=.98, bottom=.02)
    svg_path = root / 'overview.svg'
    figure.savefig(svg_path, facecolor=figure.get_facecolor(), metadata={'Date': None})
    svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines()) + '\n')
    figure.savefig(root / 'overview.png', dpi=160, facecolor=figure.get_facecolor())
    plt.close(figure)


if __name__ == '__main__':
    main()
