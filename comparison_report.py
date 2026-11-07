"""Audit completed local screens and publish a configuration-level comparison."""
import json
from decimal import Decimal
from pathlib import Path

from amc_analysis import summarize_types
from figure_style import MUTED, PAPER, RED, TEAL, canvas, label, rectangle, rule, save


def summarize(document, name, annotations):
    rows = document['results']
    if not document['complete'] or len(rows) != 3:
        raise ValueError(f'{name} is not a completed three-question screen')
    for row in rows:
        matched = row['prediction'] != 'None' and Decimal(row['prediction']) == Decimal(row['reference'])
        if matched != row['correct']:
            raise ValueError('Saved correctness disagrees with numeric prediction')
    types = summarize_types(rows, annotations)
    return dict(
        name=name, correct=sum(row['correct'] for row in rows), attempted=len(rows),
        eos_completed=sum(row['ended_with_eos'] for row in rows),
        timeouts=sum(row['time_limit_reached'] for row in rows),
        token_caps=sum(row.get('token_limit_reached', False) for row in rows),
        mean_seconds=sum(row['elapsed_seconds'] for row in rows) / len(rows),
        by_type=types,
    )


def draw(root, documents, summaries):
    figure, axis = canvas(4 + 1.25 * len(documents), 'local-student-comparison')
    label(axis, .055, .945, 'S T U D E N T   S E L E C T I O N   /   L O C A L   S C R E E N', 8, MUTED)
    label(axis, .055, .86, 'A student must first finish and solve.', 23, weight='bold')
    label(axis, .055, .797, 'Same three hard validation questions · No training · Reserved test untouched', 10, MUTED)
    rule(axis, (.055, .745), (.945, .745))
    label(axis, .055, .695, 'CONFIGURATION', 8, MUTED)
    for horizontal, number in [(.49, 17), (.65, 18), (.81, 25)]:
        label(axis, horizontal, .695, f'PROBLEM {number}', 8, MUTED, ha='center')
    for index, (document, summary) in enumerate(zip(documents, summaries)):
        step = .39 / len(documents)
        vertical = .62 - index * step
        rectangle(axis, (.055, vertical - step * .4), .89, step * .8, PAPER)
        label(axis, .07, vertical + .018, summary['name'], 10, weight='bold')
        label(axis, .07, vertical - .02,
              f"{summary['correct']}/3 numeric matches · {summary['eos_completed']}/3 EOS completions", 8.5, MUTED)
        lookup = {int(row['url'].split('_')[-1]): row for row in document['results']}
        for horizontal, number in [(.49, 17), (.65, 18), (.81, 25)]:
            row = lookup[number]
            color = TEAL if row['correct'] else (RED if row['ended_with_eos'] else MUTED)
            predicted = row['prediction'] if row['prediction'] != 'None' else '—'
            label(axis, horizontal, vertical + .014, predicted, 17, color, 'bold', ha='center')
            status = 'MATCH' if row['correct'] else ('WRONG' if row['ended_with_eos'] else 'INCOMPLETE')
            label(axis, horizontal, vertical - .024, status, 7, color, ha='center')
    label(axis, .055, .17, 'REFERENCE ANSWERS', 8, MUTED)
    reference = {int(row['url'].split('_')[-1]): row['reference'] for row in documents[0]['results']}
    for horizontal, number in [(.49, 17), (.65, 18), (.81, 25)]:
        label(axis, horizontal, .17, f'{float(reference[number]):g}', 11, weight='bold', ha='center')
    label(axis, .055, .105, 'A small configuration screen, not a statistically reliable model ranking.', 9)
    label(axis, .055, .052, 'Historical baseline: different backend, precision, prompt and limits. New candidates: 4-bit CPU MLX; different sampling settings.', 7.6, MUTED)
    save(figure, root, 'comparison')


def main():
    root = Path('amc_results')
    annotations = json.loads((root / 'problem_types.json').read_text())
    verification = json.loads((root / 'answer_verification.json').read_text())
    if not verification['all_agree']:
        raise ValueError('Independent key checks did not agree')
    documents = [json.loads((root / 'calibration.json').read_text())]
    names = ['Qwen2.5-Math 1.5B · historical']
    registry = json.loads(Path('comparison_config.json').read_text())['candidates']
    for candidate, settings in registry.items():
        documents.append(json.loads((root / 'model_comparison' / f'{candidate}.json').read_text()))
        names.append(settings['display_name'])
    summaries = [summarize(document, name, annotations) for document, name in zip(documents, names)]
    (root / 'model_comparison/summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
    table = ['# Local student screen', '',
        'Three shared validation questions, one sampled answer each. No training or final-test evaluation.', '',
        '| Configuration | Numeric matches | EOS completions | Timeouts | Token caps | Mean seconds |',
        '|---|---:|---:|---:|---:|---:|']
    for summary in summaries:
        table.append(f"| {summary['name']} | {summary['correct']}/3 | {summary['eos_completed']}/3 | {summary['timeouts']} | {summary['token_caps']} | {summary['mean_seconds']:.1f} |")
    table.extend(['', '## Question types', '',
                  '| Configuration | Counting / combinatorics | Geometry |', '|---|---:|---:|'])
    for summary in summaries:
        groups = {group['category']: group for group in summary['by_type']}
        counting, geometry = groups['Counting / combinatorics'], groups['Geometry']
        table.append(f"| {summary['name']} | {counting['correct']}/{counting['attempted']} | {geometry['correct']}/{geometry['attempted']} |")
    table.extend(['',
        'These denominators are two counting questions and one geometry question, not estimates of broad subject ability.',
        'The older baseline has different precision, backend, prompt and limits. The new candidates use different sampling settings. Timings and scores do not isolate model-size or architecture effects.',
        'A timeout is incomplete generation under this local budget, not evidence that a model cannot solve the problem. The geometry diagram is source code rather than a rendered image for every configuration.',
        'Any apparent advantage needs more validation problems and repeated samples. Unknown pretraining overlap remains a limitation.'])
    (root / 'model_comparison/RESULTS.md').write_text('\n'.join(table) + '\n')
    draw(root, documents, summaries)
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
