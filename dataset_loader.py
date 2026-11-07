"""Load a saved dataset or an explicitly referenced, checksum-verified copy."""
import hashlib
import json


def load_dataset(root):
    path = root / 'data.json'
    if not path.exists():
        reference = json.loads((root / 'dataset_reference.json').read_text())
        path = root / reference['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != reference['sha256']:
            raise ValueError('Referenced dataset checksum mismatch')
    return json.loads(path.read_text())
