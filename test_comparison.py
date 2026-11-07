import unittest
from decimal import Decimal

from comparison_utils import extract_answer


class ComparisonScoringTests(unittest.TestCase):
    def test_unfinished_reasoning_is_not_a_final_answer(self):
        self.assertIsNone(extract_answer(r'Draft: \boxed{4}', thinking=True))
        self.assertEqual(extract_answer(r'Draft \boxed{4}</think>Final \boxed{8}', thinking=True), Decimal(8))

    def test_complete_numeric_answers_only(self):
        self.assertEqual(extract_answer(r'\boxed{8.0}'), Decimal(8))
        self.assertIsNone(extract_answer('#### 8/2 = 4'))
        self.assertIsNone(extract_answer(r'\boxed{8'))


if __name__ == '__main__':
    unittest.main()
