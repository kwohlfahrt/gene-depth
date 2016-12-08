#!/usr/bin/env python3
from pathlib import Path
from tifffile import imread, imsave
from scipy.ndimage import morphology, label, filters
from numpy import ones
from functools import partial

from skimage.morphology import watershed
from skimage.filters import threshold_otsu


def local_max(data, footprint):
    return data == filters.maximum_filter(data, footprint=footprint, mode='constant')


def segment(args):
    image = imread(str(args.image))
    threshold = args.threshold if args.threshold != 'otsu' else threshold_otsu(image)
    image = image > threshold

    structure = ones(args.structure, dtype='bool')
    processes = [partial(morphology.binary_opening, structure=structure),
                 partial(morphology.binary_closing, structure=structure),
                 morphology.binary_fill_holes]
    for process in processes:
        image = process(image)

    distance = morphology.distance_transform_edt(image, args.scale)

    footprint = ones(tuple(int(args.min_size // s) for s in args.scale))
    labeled, nlabels = label(local_max(distance * (distance > args.min_size / 2), footprint))

    labeled = watershed(-distance, labeled, mask=image)
    imsave(str(args.output), labeled.astype('int32'))


def plot(args):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1)
    ax.imshow(imread(str(args.image)))
    plt.show()


def main(args=None):
    from argparse import ArgumentParser
    from sys import argv

    parser = ArgumentParser(description="Align two sets of points.")
    subparsers = parser.add_subparsers()

    segment_parser = subparsers.add_parser("segment")
    segment_parser.add_argument("image", type=Path, help="The image to segment")
    segment_parser.add_argument("output", type=Path, help="The segmented image")
    segment_parser.add_argument("--threshold", type=float, help="The background threshold.")
    segment_parser.add_argument("--scale", type=float, nargs='+',
                                help="The scale of each axis")
    segment_parser.add_argument("--structure", type=int, nargs='+',
                               help="The size of the structuring elements")
    segment_parser.add_argument("--min-size", type=float, default=5.0,
                                help="The minimum size of a region")
    segment_parser.set_defaults(func=segment)

    plot_parser = subparsers.add_parser("plot")
    plot_parser.add_argument("image", type=Path, help="The segmented image to show")
    plot_parser.set_defaults(func=plot)

    args = parser.parse_args(argv[1:] if args is None else args)
    args.func(args)


if __name__ == "__main__":
    main()
