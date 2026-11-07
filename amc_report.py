"""Draw only measured AMC validation calibration, never a training/test curve."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('amc_results')
calibration=json.loads((root/'calibration.json').read_text())
gate=json.loads((root/'GATE.json').read_text())
assert calibration['complete'] and len(calibration['results'])==3
rows=calibration['results']
assert all(row['split']=='validation' and '/2022_AMC_12B_' in row['url'] for row in rows)
figure,axes=plt.subplots(1,2,figsize=(10,4))
labels=[row['url'].split('/')[-1].replace('Problem_','P') for row in rows]
axes[0].bar(labels,[row['tokens'] for row in rows])
axes[0].axhline(calibration['max_new_tokens'],linestyle='--',color='gray',label='Configured ceiling (not necessarily reached)')
axes[0].set(ylabel='Generated tokens',title='Measured output versus candidate budget')
axes[0].legend(fontsize=7)
axes[1].bar(labels,[row['tokens_per_second'] for row in rows])
axes[1].set(ylabel='Tokens / second',title='Measured CPU generation speed')
figure.suptitle('Qwen2.5-Math-1.5B: three hard AMC12 validation problems')
figure.tight_layout()
figure.savefig(root/'calibration.svg')
figure.savefig(root/'calibration.png',dpi=160)
table=['# AMC12 local calibration measurements','',
    '**Validation only. No weights updated; no final test or three-policy study run.**','',
    '| 2022 AMC12B problem | Generated tokens | Seconds | Tokens/s | EOS completed | Correct numeric answer |',
    '|---|---:|---:|---:|---|---|']
for row in rows:
    table.append(f"| [{row['url'].split('/')[-1]}]({row['url']}) | {row['tokens']} | {row['elapsed_seconds']:.2f} | {row['tokens_per_second']:.2f} | {row['ended_with_eos']} | {row['correct']} |")
table.extend(['',f"Feasibility gate passed: **{gate['passed']}**.",
    'Correctness uses publisher-provided adapted integer keys; it is not an independent proof of solution validity.',
    'Full generations stay local under ignored work/amc. Numeric predictions, identifiers and timings are published in calibration.json.',
    'A hypothetical 2,048-token output at measured speed is an extrapolation, not an observed full completion.'])
(root/'RESULTS.md').write_text('\n'.join(table)+'\n')
print('\n'.join(table))
