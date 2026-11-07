import unittest
from decimal import Decimal
from amc import final_number, source_identity


class AMCTests(unittest.TestCase):
    def test_contest_identity(self):
        self.assertEqual(source_identity('https://artofproblemsolving.com/wiki/index.php/2022_AMC_12B_Problems/Problem_25'),(2022,'B',25))
        with self.assertRaises(ValueError):
            source_identity('https://example.com/2022_AIME_I_Problems/Problem_15')

    def test_numeric_grading(self):
        self.assertEqual(final_number(r'Work 2 + 3. Final \boxed{5}'),Decimal(5))
        self.assertEqual(final_number(r'\boxed{4} then \boxed{5}'),Decimal(5))
        self.assertIsNone(final_number('The intermediate number is 5.'))
        self.assertIsNone(final_number(r'\boxed{5/2}'))


if __name__=='__main__':
    unittest.main()
