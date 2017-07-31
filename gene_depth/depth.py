#!/usr/bin/env python3
import click

from scipy.ndimage import morphology
import numpy as np
import pandas as pd
from tifffile import imread
from itertools import repeat
from pathlib import Path
from collections import defaultdict
from operator import itemgetter

@click.group()
def depth():
    pass

def dual_distance(image, *args, **kwargs):
    distance = morphology.distance_transform_edt(image, *args, **kwargs)
    inverse_distance = morphology.distance_transform_edt(~image, *args, **kwargs)
    distance[~image] = -inverse_distance[~image]
    return distance


def load_spots(path):
    if path.suffix == ".pickle":
        with path.open("rb") as f:
            return load(f)
    elif path.suffix == ".csv":
        return np.loadtxt(str(path), ndmin=2, dtype='uint', delimiter=',')
    elif path.suffix == ".txt":
        return np.loadtxt(str(path), ndmin=2, dtype='uint')
    else:
        raise ValueError("Unrecognized file type: {}".format(path.suffix))

@depth.command()
@click.argument('label', type=str)
@click.argument('segments', type=Path)
@click.argument('peaks', type=Path)
@click.option('--mode', type=click.Choice(['absolute', 'relative']), default='absolute')
@click.option('--scale', type=(float, float), default=(1.0, 1.0),
              help="The scale along each axis")
def calc(label, segments, peaks, mode='absolute', scale=1.0):
    from sys import stdout

    segments = imread(str(segments))
    peaks = load_spots(peaks)

    data = defaultdict(list)

    for segment in np.unique(segments)[1:]:
        background = segments == segment
        distance = dual_distance(background, sampling=scale)

        cell_depths = distance[tuple(peaks[:, ::-1].T)]
        cell_depths = cell_depths[cell_depths >= 0.0]

        if len(cell_depths) != 1:
            continue

        data['gene'].extend([label] * len(cell_depths))
        if mode == 'absolute':
            data['depth'].extend(cell_depths)
        else:
            data['depth'].extend(cell_depths / distance.max() * 100)
    pd.DataFrame(data, columns=['gene', 'depth']).to_csv(stdout, index=False)


@depth.command()
@click.argument('depths', type=Path, nargs=-1)
@click.option('--unit', type=str, required=True, help="The unit for the y-axis")
@click.option('--output', type=Path)
@click.option('--figsize', type=(float, float), default=(3.0, 6.0))
def plot(depths, unit, output=None, figsize=(3.0, 6.0)):
    import matplotlib
    if output is not None:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    data = pd.concat(map(pd.read_csv, map(str, depths)))
    ax.set_ylabel("Depth ({})".format(unit))
    grouped = data.groupby(['gene'], sort=False)
    keys, values = zip(*sorted(grouped, key=itemgetter(0)))
    labels = list(map("{}\n(n={})".format, keys, map(len, values)))
    ax.boxplot([v['depth'] for v in values], labels=labels)
    if unit == '%':
        ax.set_ylim([0, 100])
    plt.setp(ax.get_xticklabels(), rotation=90)

    fig.tight_layout()
    if output is not None:
        fig.savefig(str(output))
    else:
        plt.show()
