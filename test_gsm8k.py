import unittest
from decimal import Decimal
from gsm8k import extract, prompt, training_tensors


class ScoringTests(unittest.TestCase):
    def test_explicit_answers(self):
        for text in ['#### 1,200', r'\boxed{1200}', 'The final answer is 1200', '1200.0']:
            self.assertEqual(extract(text), Decimal('1200'))
        self.assertEqual(extract('#### -2.5'), Decimal('-2.5'))

    def test_no_incidental_number(self):
        self.assertIsNone(extract('There are 5 apples and 3 pears.'))
        self.assertIsNone(extract('I cannot solve this'))
        self.assertIsNone(extract('##### 11. A new question'))
        self.assertIsNone(extract('#### 48/8 = 6'))
        self.assertIsNone(extract('#### 100% (4 ratings)'))
        self.assertNotIn('gold', prompt('How many apples?'))

    def test_solution_is_complete_and_prompt_is_masked(self):
        encoded = training_tensors({'prefix':[11,12], 'target':[21,22,2]}, 2)
        self.assertEqual(encoded['labels'][0,:6].tolist(), [-100,-100,21,22,2,-100])
        self.assertEqual(encoded['attention_mask'].sum().item(), 5)
        self.assertEqual(encoded['labels'].ne(-100).sum().item(),3)
        with self.assertRaises(ValueError):
            training_tensors({'prefix':[11]*250,'target':[21]*7},2)


if __name__ == '__main__':
    unittest.main()
