"""Independently check the three validation keys using finite enumeration and exact area."""
import json
import re
from pathlib import Path

from comparison_utils import frozen_validation


def count_binary_arrays():
    count = 0
    for mask in range(1 << 16):
        row_totals = [bin((mask >> (4 * row)) & 15).count('1') for row in range(4)]
        if sorted(row_totals) != [1, 2, 3, 4]:
            continue
        column_totals = [sum((mask >> (4 * row + column)) & 1 for row in range(4))
                         for column in range(4)]
        count += sorted(column_totals) == [1, 2, 3, 4]
    return count


def count_grid_configurations():
    count = 0
    positions = [(row, column) for row in range(1, 4) for column in range(1, 4)]
    for mask in range(1 << 9):
        filled = {position for index, position in enumerate(positions) if mask & (1 << index)}
        updated = set()
        for row in range(5):
            for column in range(5):
                neighbors = sum(
                    (row + delta_row, column + delta_column) in filled
                    for delta_row in [-1, 0, 1] for delta_column in [-1, 0, 1]
                    if delta_row != 0 or delta_column != 0
                )
                if neighbors == 3 or (neighbors == 2 and (row, column) in filled):
                    updated.add((row, column))
        count += updated == {(2, 2)}
    return count


def polygon_area(vertices):
    from sympy import Rational, simplify
    return simplify(Rational(1, 2) * abs(sum(
        start[0] * end[1] - start[1] * end[0]
        for start, end in zip(vertices, vertices[1:] + vertices[:1])
    )))


def geometry_key(problem):
    from sympy import expand, sqrt, sympify
    outline = problem.rsplit('draw(', 1)[-1].split('--cycle')[0].replace('sqrt(3)', 's')
    pairs = re.findall(r'\(([^(),]+),([^(),]+)\)', outline)
    if len(pairs) != 12:
        raise ValueError('Expected the twelve-vertex outer boundary from the supplied diagram code')
    vertices = []
    for pair in pairs:
        if not all(re.fullmatch(r'[0-9s+\-\s]+', expression) for expression in pair):
            raise ValueError('Unexpected coordinate expression')
        vertices.append(tuple(sympify(expression, locals={'s': sqrt(3)}) for expression in pair))
    area = expand(polygon_area(vertices))
    coefficient = area.coeff(sqrt(3))
    constant = expand(area - coefficient * sqrt(3))
    if not coefficient.is_Integer or not constant.is_Integer:
        raise ValueError('Area did not match the required integer-coefficient form')
    return abs(int(coefficient) + 3 + int(constant)), str(area)


def main():
    root = Path(__file__).resolve().parent
    rows = frozen_validation(root)
    full_count = count_binary_arrays()
    grid_count = count_grid_configurations()
    checks = []
    for row in rows:
        if row['number'] == 17:
            answer, method = full_count % 100, 'Enumerate all 65,536 binary 4-by-4 arrays; apply row/column constraints; take remainder modulo 100'
        elif row['number'] == 18:
            answer, method = grid_count, 'Enumerate all 512 initial 3-by-3 configurations and update the entire 5-by-5 grid once'
        else:
            answer, area = geometry_key(row['problem'])
            method = 'Exact shoelace area from the twelve-vertex boundary in the supplied diagram source; no model reasoning used'
        checks.append(dict(url=row['url'], digest=row['digest'], computed_answer=answer,
                           dataset_answer=row['answer'], agrees=answer == row['answer'], method=method))
        if row['number'] == 17:
            checks[-1]['valid_arrays_before_modulo'] = full_count
        if row['number'] == 25:
            checks[-1]['exact_area'] = area
    result = dict(scope='Only the three frozen validation problems; no reserved test inputs used',
                  limitations='Checks the stated constraints and supplied boundary coordinates; does not prove original exam transcription or diagram construction is correct',
                  all_agree=all(row['agrees'] for row in checks), checks=checks)
    (root / 'amc_results/answer_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
