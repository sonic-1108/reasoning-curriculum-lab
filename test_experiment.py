import unittest
from experiment import dataset


class DataTests(unittest.TestCase):
    def test_disjoint_and_heldout_family(self):
        rows = dataset()
        self.assertEqual(len(rows), len({row['id'] for row in rows}))
        self.assertTrue(all(row['split'] == 'ood' for row in rows if row['family'] == 'compose'))
        self.assertFalse(any(row['family'] == 'compose' for row in rows if row['split'] == 'train'))
        self.assertEqual(set(row['family'] for row in rows if row['split'] == 'train'), {'add', 'subtract', 'multiply'})


if __name__ == '__main__':
    unittest.main()
