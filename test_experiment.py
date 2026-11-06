import unittest
from experiment import dataset


class DataTests(unittest.TestCase):
    def test_disjoint_and_heldout_family(self):
        rows = dataset()
        self.assertEqual(len(rows), len({row['id'] for row in rows}))
        self.assertTrue(all(row['split'] == 'ood' for row in rows if row['family'] == 'compose'))
        self.assertFalse(any(row['family'] == 'compose' for row in rows if row['split'] == 'train'))
        self.assertEqual(set(row['family'] for row in rows if row['split'] == 'train'), {'add', 'subtract', 'multiply'})

    def test_arithmetic_and_stable_splits(self):
        rows = dataset()
        self.assertEqual(rows, dataset())
        for row in rows:
            expression = row['prompt'].removeprefix('Calculate ').removesuffix('. Answer:')
            self.assertEqual(str(eval(expression, {'__builtins__': {}})), row['answer'])
        train = {row['id'] for row in rows if row['split'] == 'train'}
        evaluation = {row['id'] for row in rows if row['split'] != 'train'}
        self.assertFalse(train & evaluation)


if __name__ == '__main__':
    unittest.main()
