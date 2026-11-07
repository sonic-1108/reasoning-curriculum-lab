import unittest
from decimal import Decimal

from comparison_utils import BudgetExceeded, check_budget, extract_answer


class ComparisonScoringTests(unittest.TestCase):
    def test_unfinished_reasoning_is_not_a_final_answer(self):
        self.assertIsNone(extract_answer(r'Draft: \boxed{4}', thinking=True))
        self.assertEqual(extract_answer(r'Draft \boxed{4}</think>Final \boxed{8}', thinking=True), Decimal(8))

    def test_complete_numeric_answers_only(self):
        self.assertEqual(extract_answer(r'\boxed{8.0}'), Decimal(8))
        self.assertIsNone(extract_answer('#### 8/2 = 4'))
        self.assertIsNone(extract_answer(r'\boxed{8'))
        self.assertIsNone(extract_answer(r'<think>Draft \boxed{8}', thinking=False))

    def test_exact_area_uses_all_vertices(self):
        from verify_amc import polygon_area
        self.assertEqual(polygon_area([(0, 0), (2, 0), (2, 3), (0, 3)]), 6)
        self.assertEqual(polygon_area([(0, 3), (2, 3), (2, 0), (0, 0)]), 6)

    def test_deadline_includes_prompt_processing(self):
        check_budget(10, 300, 309.9)
        with self.assertRaises(BudgetExceeded):
            check_budget(10, 300, 310)


if __name__ == '__main__':
    unittest.main()
