import copy
import json
import unittest
from collections import Counter
from pathlib import Path

from study_matrix import build_plan


class StudyDesignTests(unittest.TestCase):
    def setUp(self):
        self.candidates = json.loads((Path(__file__).parent / 'comparison_config.json').read_text())['candidates']

    def test_balanced_conditions_preserve_each_models_initial_checkpoint(self):
        plan = build_plan(self.candidates)
        self.assertEqual(len(plan['runs']), len(self.candidates) * 9)
        combinations = {(row['candidate'], row['seed'], row['policy']) for row in plan['runs']}
        self.assertEqual(len(combinations), len(plan['runs']))
        self.assertEqual(set(Counter((row['candidate'], row['policy']) for row in plan['runs']).values()), {3})
        for row in plan['runs']:
            expected = self.candidates[row['candidate']]
            self.assertEqual(row['initial_revision'], expected['revision'])
            self.assertEqual(row['initial_weights_sha256'], expected['weights_sha256'])
            self.assertEqual(row['status'], 'proposed_not_run')
            self.assertIsNone(row['supervised_token_budget'])

    def test_same_lineage_and_repeated_seeds_do_not_pass_design_checks(self):
        same_lineage = copy.deepcopy(self.candidates)
        for entry in same_lineage.values():
            entry['pretraining_lineage'] = 'Qwen'
        with self.assertRaises(ValueError):
            build_plan(same_lineage)
        with self.assertRaises(ValueError):
            build_plan(self.candidates, seeds=(7, 7, 41))


if __name__ == '__main__':
    unittest.main()
