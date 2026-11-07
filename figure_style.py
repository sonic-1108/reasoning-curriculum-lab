"""Shared publication styling for the study's data-driven figures."""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

INK = '#1c2b33'
MUTED = '#697980'
RULE = '#dce3e5'
PAPER = '#f3f6f6'
TEAL = '#287f77'
RED = '#b45049'


def canvas(height, identity):
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'svg.fonttype': 'none',
        'svg.hashsalt': identity,
    })
    figure = plt.figure(figsize=(12, height), facecolor='white')
    axis = figure.add_axes((0, 0, 1, 1))
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis('off')
    return figure, axis


def label(axis, horizontal, vertical, text, size=10, color=INK, weight='normal', **options):
    axis.text(horizontal, vertical, text, fontsize=size, color=color,
              fontweight=weight, va='center', **options)


def rule(axis, start, end, color=RULE, width=.8, **options):
    axis.plot([start[0], end[0]], [start[1], end[1]],
              color=color, linewidth=width, **options)


def rectangle(axis, position, width, height, color):
    axis.add_patch(Rectangle(position, width, height, facecolor=color, edgecolor='none'))


def save(figure, root, name):
    path = root / f'{name}.svg'
    figure.savefig(path, metadata={'Date': None})
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    figure.savefig(root / f'{name}.png', dpi=180)
    plt.close(figure)
