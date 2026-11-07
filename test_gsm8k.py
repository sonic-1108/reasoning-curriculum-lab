import unittest
from decimal import Decimal
from gsm8k import extract, prompt


class ScoringTests(unittest.TestCase):
    def test_explicit_answers(self):
        for text in ['#### 1,200', r'\boxed{1200}', 'The final answer is 1200', '1200.0']:
            self.assertEqual(extract(text), Decimal('1200'))
        self.assertEqual(extract('#### -2.5'), Decimal('-2.5'))

    def test_no_incidental_number(self):
        self.assertIsNone(extract('There are 5 apples and 3 pears.'))
        self.assertIsNone(extract('I cannot solve this'))
        self.assertNotIn('gold', prompt('How many apples?'))


if __name__ == '__main__':
    unittest.main()
