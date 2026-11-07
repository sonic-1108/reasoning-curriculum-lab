"""Summarize descriptive problem types without inferring unsupported rankings."""


def summarize_types(rows, annotations):
    labels = {entry['url']: entry for entry in annotations['problems']}
    if len(labels) != len(annotations['problems']):
        raise ValueError('Duplicate problem annotation')
    if len({row['url'] for row in rows}) != len(rows):
        raise ValueError('Repeated validation problem')
    if {row['url'] for row in rows} != set(labels):
        raise ValueError('Annotations must match the evaluated problems exactly')
    groups = {}
    for row in rows:
        annotation = labels[row['url']]
        if annotation['digest'] != row['digest']:
            raise ValueError('Problem text changed after annotation')
        if not isinstance(row['correct'], bool):
            raise ValueError('Correctness must be boolean')
        category = annotation['category']
        group = groups.setdefault(category, {'category': category, 'attempted': 0, 'correct': 0})
        group['attempted'] += 1
        group['correct'] += int(row['correct'])
    return [dict(group, accuracy=group['correct'] / group['attempted'])
            for group in sorted(groups.values(), key=lambda group: group['category'])]
