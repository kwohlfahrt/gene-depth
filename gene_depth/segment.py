#!/usr/bin/env python3
import click

from pathlib import Path
from tifffile import imread, imsave
from scipy.ndimage import morphology, label, filters
import numpy as np
from functools import partial

from skimage.morphology import watershed
from skimage.filters import threshold_otsu

@click.group()
def segment():
    pass


def local_max(data, footprint):
    return data == filters.maximum_filter(data, footprint=footprint, mode='constant')


@segment.command()
@click.argument('image', type=Path)
@click.argument('output', type=Path)
@click.option('--threshold', type=float, default=None,
              help='The background threshold')
@click.option('--scale', type=(float, float), default=(1.0, 1.0),
              help='The scale along each axis')
@click.option('--structure', type=(int, int), default=(3, 3),
              help="The size of the structuring element.")
@click.option('--min-size', type=float, default=5.0,
              help="The minimum size of a region")
def run(image, output, threshold, scale=(1.0, 1.0), structure=(1, 1), min_size=5.0):
    image = imread(str(image))
    threshold = threshold if threshold != None else threshold_otsu(image)
    image = image > threshold

    structure = np.ones(structure, dtype='bool')
    processes = [partial(morphology.binary_opening, structure=structure),
                 partial(morphology.binary_closing, structure=structure),
                 morphology.binary_fill_holes]
    for process in processes:
        image = process(image)

    distance = morphology.distance_transform_edt(image, scale)

    footprint = np.ones(tuple(int(min_size // s) for s in scale))
    labeled, nlabels = label(local_max(distance, footprint) & (distance > min_size / 2))

    labeled = watershed(-distance, labeled, mask=image)
    imsave(str(output), labeled.astype('int32'))


@segment.command()
@click.argument("image", type=Path)
@click.option('--output', type=Path)
def plot(image, output=None):
    import matplotlib
    from matplotlib.colors import hsv_to_rgb
    if output is not None:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rng = np.random.RandomState(4)

    image = imread(str(image))

    ncolors = image.max()
    colors = np.ones((ncolors+1, 3), dtype='float')
    colors[1:, 1] = 0.9 # saturation
    colors[1:, 2] = 0.95 # value
    colors[1:, 0] = rng.uniform(0, 1, size=ncolors)
    colors[0, :] = [0, 0, 1] # white background
    colors = hsv_to_rgb(colors)

    fig, ax = plt.subplots(1, 1)
    ax.imshow(colors[image], interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    if output is not None:
        fig.savefig(str(output))
    else:
        plt.show()
