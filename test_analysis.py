import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from amc_analysis import summarize_types
from dataset_loader import load_dataset


class EvidenceTests(unittest.TestCase):
    def test_category_denominators_and_correctness(self):
        rows = [
            {'url': 'one', 'digest': 'a', 'correct': True},
            {'url': 'two', 'digest': 'b', 'correct': False},
            {'url': 'three', 'digest': 'c', 'correct': False},
        ]
        annotations = {'problems': [
            {'url': 'one', 'digest': 'a', 'category': 'Counting'},
            {'url': 'two', 'digest': 'b', 'category': 'Counting'},
            {'url': 'three', 'digest': 'c', 'category': 'Geometry'},
        ]}
        result = summarize_types(rows, annotations)
        self.assertEqual(result[0], {'category': 'Counting', 'attempted': 2, 'correct': 1, 'accuracy': .5})
        self.assertEqual(result[1]['accuracy'], 0)
        with self.assertRaises(ValueError):
            summarize_types(rows + [rows[0]], annotations)
        annotations['problems'][0]['digest'] = 'changed'
        with self.assertRaises(ValueError):
            summarize_types(rows, annotations)

    def test_dataset_reference_rejects_changed_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shared = root / 'shared.json'
            shared.write_text('[1, 2, 3]')
            reference = {'path': 'shared.json', 'sha256': hashlib.sha256(shared.read_bytes()).hexdigest()}
            (root / 'dataset_reference.json').write_text(json.dumps(reference))
            self.assertEqual(load_dataset(root), [1, 2, 3])
            shared.write_text('[4, 5, 6]')
            with self.assertRaises(ValueError):
                load_dataset(root)


if __name__ == '__main__':
    unittest.main()
