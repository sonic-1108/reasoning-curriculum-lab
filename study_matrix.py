"""Build a proposed cross-model study matrix; this script does not train models."""
import json
from pathlib import Path

POLICIES = ('random', 'easy_to_hard', 'adaptive')


def build_plan(candidates, seeds=(7, 19, 41)):
    if len(candidates) < 3:
        raise ValueError('The proposed study requires at least three student candidates')
    if len({entry['pretraining_lineage'] for entry in candidates.values()}) < 2:
        raise ValueError('Multiple model names from one lineage do not establish cross-family coverage')
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError('At least three distinct training seeds are required')
    runs = []
    for candidate, settings in candidates.items():
        for seed in seeds:
            for policy in POLICIES:
                runs.append(dict(
                    candidate=candidate, policy=policy, seed=seed,
                    initial_model=settings['model'], initial_revision=settings['revision'],
                    initial_weights_sha256=settings['weights_sha256'],
                    supervised_token_budget=None, status='proposed_not_run',
                ))
    return dict(
        purpose='Cross-model method validation; separate from untrained-student screening',
        status='Planning only; no hard-math training runs are implemented or completed by this generator',
        readiness='Blocked until students pass validation, trusted training solutions are available, and budgets are frozen',
        required_viable_students=3, required_pretraining_lineages=2,
        candidate_count=len(candidates), pretraining_lineage_count=len({entry['pretraining_lineage'] for entry in candidates.values()}),
        planned_runs_if_all_candidates_pass=len(runs), seeds=list(seeds), policies=list(POLICIES),
        primary_effect='Within each model: accuracy after adaptive training minus accuracy after random training',
        secondary_effect='Within each model: accuracy after adaptive training minus accuracy after fixed-curriculum training',
        baseline='Evaluate each initial checkpoint before any experiment-specific training',
        runs=runs,
    )


def main():
    root = Path(__file__).resolve().parent
    candidates = json.loads((root / 'comparison_config.json').read_text())['candidates']
    plan = build_plan(candidates)
    (root / 'study_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(f"PROPOSED ONLY: {plan['candidate_count']} candidates, {len(POLICIES)} policies, {len(plan['seeds'])} seeds; {len(plan['runs'])} conditional training runs")


if __name__ == '__main__':
    main()
