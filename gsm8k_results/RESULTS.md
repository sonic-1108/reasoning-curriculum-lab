# GSM8K fixed-subset measurements

24 shared test problems; counts across seeds are repeated observations, not independent problems.

| Policy | Final answer accuracy | Correct by seed | Valid final answer | Token cap reached | Train seconds | Select seconds |
|---|---:|---|---:|---:|---:|---:|
| base | 0.0% | {'0': 0} | 8.3% | 95.8% | 0.00 | 0.00 |
| random | 0.0% | {'7': 0, '19': 0, '41': 0} | 12.5% | 97.2% | 6.82 | 0.00 |
| easy_to_hard | 1.4% | {'7': 0, '19': 0, '41': 1} | 13.9% | 95.8% | 6.83 | 0.00 |
| adaptive | 2.8% | {'7': 1, '19': 1, '41': 0} | 19.4% | 95.8% | 6.88 | 6.41 |

Training and selection seconds are means across seeds; evaluation time is saved separately in costs.json.
Correct final numbers do not certify correct explanations. Some matches occur with invalid reasoning or at the generation limit.

Supervised tokens per seed (identical across policies):
{'7': 958, '19': 1165, '41': 1221}
